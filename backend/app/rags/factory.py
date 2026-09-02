from __future__ import annotations

from app.core.config import Settings
from app.core.llama_client import LlamaClients
from app.rags.base import RAGBase
from app.rags.graph import GraphRAG
from app.rags.relational import RelationalRAG
from app.rags.vectorial import VectorialRAG


async def build_rags(clients: LlamaClients, settings: Settings) -> dict[str, RAGBase]:
    """The three backends, keyed by the router's route names.

    Built in one place so the ingestion path and the search path can share the
    same instances. GraphRAG keeps the graph in memory, so two separate copies
    would mean nothing ingested is ever visible to a search until restart.
    """
    return {
        "relational": RelationalRAG(clients, settings),
        "vectorial": await VectorialRAG.create(clients, settings),
        "graph": GraphRAG.load(clients, settings),
    }
