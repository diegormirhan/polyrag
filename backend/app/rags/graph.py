from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import networkx as nx
from opentelemetry import trace

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, chat, embed
from app.core.vectors import dot, normalize
from app.ingest.chunking import TRAIL_SEPARATOR
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


def section_headings(texts: Iterable[str]) -> list[str]:
    """The section each chunk belongs to, deduplicated — the store's own vocabulary.

    Chunking prefixes every chunk with its heading trail, so the first line already
    says where the text came from. Only the deepest level is kept: the document
    title names a file rather than a subject, the same title can head chunks that
    ended up in different stores, and using it would teach the router that both
    stores hold the same thing.
    """
    headings = set()
    for text in texts:
        first = text.split("\n", 1)[0].strip()
        # A bare "# Title" line is a document title, not a section.
        if not first or first.startswith("#"):
            continue
        headings.add(first.split(TRAIL_SEPARATOR)[-1].strip())
    return sorted(h for h in headings if h)


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

    async def ingest(self, content: str) -> bool:
        """Stores the chunk and its triples. False when there was nothing to store.

        The caller needs the answer. Text with no extractable relation is not graph
        material, and silently returning used to drop it from every store at once:
        measured on the demo corpus, 4 of 14 chunks routed here existed in no store
        afterwards, retrievable by nothing.
        """
        triples = await self._extract_triples(content)
        if not triples:
            return False

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
        return True

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

    async def content_anchors(self) -> list[str]:
        return section_headings(
            data["text"] for _, data in self._graph.nodes(data=True) if data.get("kind") == "chunk"
        )

    def _entities(self) -> list[str]:
        return [n for n, d in self._graph.nodes(data=True) if d.get("kind") == "entity"]

    async def _seed_entities(self, question: str) -> dict[str, float]:
        """The nodes to start the walk from, and how hard each one pulls.

        Seeded by comparing the whole question against every entity name, not by
        running named-entity extraction on the question first. NER was measured
        returning an empty list for 6 of the 16 graph questions in the golden set,
        and `query` then retrieved nothing at all. Every one of those six describes
        what it wants instead of naming it -- "which rule governs personal data",
        "who approves a thirty-thousand purchase" -- which is exactly the shape a
        knowledge graph should be good at. The entity that would answer is the
        answer, so it is not in the question to be extracted.

        Matching the question itself found the right node in all 16, and removed a
        model call from the search path: seeding is now pure arithmetic.

        Weighted by similarity rather than uniformly, which is also what the
        HippoRAG personalisation vector expects: a node matching at 0.70 should
        pull harder than one at 0.46.
        """
        # An explicit `with`, not the decorator: as a decorator on an async function
        # the span opens when the coroutine object is created rather than when the
        # body runs, and the recorded start time then precedes its own parent's.
        with _tracer.start_as_current_span("rag.graph.seeds") as span:
            graph_entities = self._entities()
            # graph.entities is on the span on purpose: this step embeds EVERY entity
            # in the graph on every query, so the attribute is the early warning for
            # a cost that grows with the corpus.
            span.set_attribute("graph.entities", len(graph_entities))
            if not graph_entities:
                return {}

            cfg = self._settings.rags.graph
            raw = await embed(self._clients.embeddings, [question, *graph_entities])
            question_vector, *entity_vectors = [normalize(v) for v in raw]

            pairs = zip(graph_entities, entity_vectors, strict=True)
            scored = sorted(((dot(question_vector, v), name) for name, v in pairs), reverse=True)
            # The floor keeps an unrelated question from seeding the walk with the
            # least-bad node it can find and answering confidently from nothing.
            # Measured on this corpus: off-topic questions peak at 0.41, real graph
            # questions start at 0.53, so 0.45 has margin on both sides.
            seeds = {name: score for score, name in scored[: cfg.seed_top_k] if score >= cfg.seed_floor}
            span.set_attributes(
                {"graph.seeds_matched": len(seeds), "graph.seed_best": scored[0][0] if scored else 0.0}
            )
            return seeds

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
                personalization=seeds,
            )
        chunk_ids = [n for n in scores if self._graph.nodes[n].get("kind") == "chunk"]
        if not chunk_ids:
            return []
        ranked = await self._fuse(question, chunk_ids, scores)
        return [
            {
                "text": self._graph.nodes[node]["text"],
                "score": score,
                "seeds": list(seeds),
                "pagerank": scores[node],
            }
            for node, score in ranked[:top_k]
        ]

    async def _fuse(
        self, question: str, chunk_ids: list[str], pagerank: dict[str, float]
    ) -> list[tuple[str, float]]:
        """Ranks chunks by fusing the PageRank order with direct question similarity.

        PageRank answers "what is this question's neighbourhood in the graph" and is
        the only thing here that can reach a chunk sharing no words with the
        question. It is not good at choosing between several chunks that all mention
        the same entity, because it ranks them by connectivity: measured, "Quem pode
        suspender o Contrato Marco 2026" put the chunk *about* that contract first
        and the chunk that names who can suspend it second.

        Cosine between the question and the chunk text answers exactly that second
        question, and is blind to the first. On the 16 graph questions of the golden
        set, PageRank alone puts the right chunk first 66% of the time and cosine
        alone 88%, but they fail on different questions -- which is the condition
        under which fusing beats either input.

        Reciprocal Rank Fusion because the two scores are not comparable: PageRank
        is a probability over a walk, cosine is an angle. RRF uses only the ranks,
        so no scaling has to be invented, and it is pure arithmetic.
        """
        with _tracer.start_as_current_span("rag.graph.fuse") as span:
            texts = [self._graph.nodes[node]["text"] for node in chunk_ids]
            vectors = [normalize(v) for v in await embed(self._clients.embeddings, [question, *texts])]
            question_vector, chunk_vectors = vectors[0], vectors[1:]

            similarity = {
                node: dot(question_vector, vector)
                for node, vector in zip(chunk_ids, chunk_vectors, strict=True)
            }
            by_pagerank = sorted(chunk_ids, key=lambda n: pagerank[n], reverse=True)
            by_cosine = sorted(chunk_ids, key=lambda n: similarity[n], reverse=True)
            k = self._settings.rags.graph.rrf_k
            positions = (
                {node: index for index, node in enumerate(by_pagerank, start=1)},
                {node: index for index, node in enumerate(by_cosine, start=1)},
            )
            fused = {node: sum(1 / (k + rank[node]) for rank in positions) for node in chunk_ids}
            span.set_attribute("graph.fused_chunks", len(fused))
            return sorted(fused.items(), key=lambda item: item[1], reverse=True)
