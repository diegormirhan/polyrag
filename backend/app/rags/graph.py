from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
from opentelemetry import trace

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, chat, embed
from app.core.vectors import dot, normalize
from app.rags.base import RAGBase, content_id

_tracer = trace.get_tracer("polyrag.rag")

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_LEADING_ARTICLE = re.compile(r"^(?:o|a|os|as|um|uma)\s+", re.IGNORECASE)


def normalize_entity(name: str) -> str:
    """Canonical form of an entity name, so the same thing becomes the same node.

    The LLM is inconsistent across calls ("A Fabrica Beta" in one chunk,
    "Fabrica_Beta" in the next). Without this, the graph fragments into
    disconnected islands and multi-hop retrieval silently stops working.
    """
    return _LEADING_ARTICLE.sub("", name.replace("_", " ").strip()).strip()


def _load_json_array(raw: str) -> list:
    cleaned = _JSON_FENCE.sub("", raw).strip()
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        # A malformed reply means no result for this text, not a crashed ingest.
        return []
    return items if isinstance(items, list) else []


def parse_triples(raw: str, max_entity_words: int = 5) -> list[tuple[str, str, str]]:
    """Parses the LLM's JSON array into (subject, relation, object) tuples.

    Triples whose subject or object runs past max_entity_words are dropped: a name
    that long is a clause, not an entity, and it becomes a graph node nothing else
    will ever match ("anonimizacao antes de qualquer exportacao para terceiros" was
    a real one). Enforced here rather than trusted to the prompt — a 4B model
    ignores "at most N words" often enough to matter, and a length limit is the
    kind of constraint that should be arithmetic, not hope.
    """
    triples = (
        (normalize_entity(item["subject"]), item["relation"], normalize_entity(item["object"]))
        for item in _load_json_array(raw)
        if isinstance(item, dict) and {"subject", "relation", "object"} <= item.keys()
    )
    return [
        (subject, relation, obj)
        for subject, relation, obj in triples
        if _is_entity_name(subject, max_entity_words) and _is_entity_name(obj, max_entity_words)
    ]


def _is_entity_name(name: str, max_words: int) -> bool:
    return bool(name) and len(name.split()) <= max_words


def parse_entities(raw: str) -> list[str]:
    """Parses the LLM's JSON array of entity names (NER on the user's question)."""
    return [
        normalize_entity(item) for item in _load_json_array(raw) if isinstance(item, str) and item.strip()
    ]


class GraphRAG(RAGBase):
    """RAG 3 — HippoRAG 2 reimplemented: OpenIE triples -> networkx -> Personalized PageRank."""

    def __init__(self, graph: nx.DiGraph, clients: LlamaClients, settings: Settings) -> None:
        self._graph = graph
        self._clients = clients
        self._settings = settings
        self._store_path = Path(settings.paths.graph_store)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, clients: LlamaClients, settings: Settings | None = None) -> GraphRAG:
        settings = settings or load_config()
        path = Path(settings.paths.graph_store)
        if path.exists():
            graph = nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")), edges="edges")
        else:
            graph = nx.DiGraph()
        return cls(graph, clients, settings)

    def save(self) -> None:
        data = nx.node_link_data(self._graph, edges="edges")
        self._store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        prompt = self._settings.rags.graph.openie_prompt.format(text=text)
        raw = await chat(self._clients.llm, [{"role": "user", "content": prompt}], temperature=0)
        return parse_triples(raw, self._settings.rags.graph.max_entity_words)

    async def _extract_entities(self, question: str) -> list[str]:
        # Questions get NER, not OpenIE: a question asserts no fact, so asking for
        # triples returns an empty list (the model correctly follows "do not infer").
        prompt = self._settings.rags.graph.ner_prompt.format(text=question)
        raw = await chat(self._clients.llm, [{"role": "user", "content": prompt}], temperature=0)
        return parse_entities(raw)

    async def ingest(self, content: str) -> None:
        triples = await self._extract_triples(content)
        if not triples:
            return

        chunk_id = f"chunk_{content_id(content)}"
        self._graph.add_node(chunk_id, kind="chunk", text=content)

        for subject, relation, obj in triples:
            self._graph.add_node(subject, kind="entity")
            self._graph.add_node(obj, kind="entity")
            self._graph.add_edge(subject, obj, relation=relation)
            # Entity -> chunk edges are what let PageRank score flow from the
            # question's entities back to retrievable text.
            self._graph.add_edge(subject, chunk_id, relation="mentioned_in")
            self._graph.add_edge(obj, chunk_id, relation="mentioned_in")

        self.save()

    async def stats(self) -> dict:
        kinds = [data.get("kind") for _, data in self._graph.nodes(data=True)]
        relations = sum(
            1 for _, _, data in self._graph.edges(data=True) if data.get("relation") != "mentioned_in"
        )
        return {
            "entities": kinds.count("entity"),
            "chunks": kinds.count("chunk"),
            "relations": relations,
        }

    def _entities(self) -> list[str]:
        return [n for n, d in self._graph.nodes(data=True) if d.get("kind") == "entity"]

    async def _seed_entities(self, question: str) -> list[str]:
        # An explicit `with`, not the decorator: as a decorator on an async function
        # the span opens when the coroutine object is created rather than when the
        # body runs, and the recorded start time then precedes its own parent's.
        with _tracer.start_as_current_span("rag.graph.seeds") as span:
            question_entities = set(await self._extract_entities(question))
            graph_entities = self._entities()
            # graph.entities is on the span on purpose: this step embeds EVERY entity
            # in the graph on every query, so the attribute is the early warning for
            # a cost that grows with the corpus.
            span.set_attributes(
                {
                    "graph.entities": len(graph_entities),
                    "graph.question_entities": len(question_entities),
                }
            )
            if not question_entities or not graph_entities:
                return []

            names = list(question_entities) + graph_entities
            vectors = [normalize(v) for v in await embed(self._clients.embeddings, names)]
            question_vectors = vectors[: len(question_entities)]
            graph_vectors = vectors[len(question_entities) :]

            threshold = self._settings.rags.graph.entity_match_threshold
            seeds = set()
            for q_vector in question_vectors:
                for name, g_vector in zip(graph_entities, graph_vectors, strict=True):
                    if dot(q_vector, g_vector) >= threshold:
                        seeds.add(name)
            span.set_attribute("graph.seeds_matched", len(seeds))
            return list(seeds)

    async def query(self, question: str, top_k: int = 5) -> list[dict]:
        seeds = await self._seed_entities(question)
        if not seeds:
            return []

        # Personalized PageRank: the random walk always teleports back to the
        # question's entities, so scores mean "relevant to THIS question".
        with _tracer.start_as_current_span("rag.graph.pagerank") as span:
            span.set_attributes(
                {
                    "graph.nodes": self._graph.number_of_nodes(),
                    "graph.edges": self._graph.number_of_edges(),
                    "graph.seeds": len(seeds),
                }
            )
            scores = nx.pagerank(
                self._graph,
                alpha=self._settings.rags.graph.pagerank_damping,
                personalization=dict.fromkeys(seeds, 1.0),
            )
        chunks = [
            {"text": self._graph.nodes[node]["text"], "score": score, "seeds": seeds}
            for node, score in scores.items()
            if self._graph.nodes[node].get("kind") == "chunk"
        ]
        return sorted(chunks, key=lambda c: c["score"], reverse=True)[:top_k]
