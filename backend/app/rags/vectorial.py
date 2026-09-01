from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import Settings, load_config
from app.core.llama_client import LlamaClients, embed
from app.rags.base import RAGBase


class VectorialRAG(RAGBase):
    """RAG 2 — free text embedded into Qdrant, retrieved by cosine/HNSW."""

    def __init__(self, client: AsyncQdrantClient, clients: LlamaClients, settings: Settings) -> None:
        self._client = client
        self._clients = clients
        self._settings = settings

    @classmethod
    async def create(cls, clients: LlamaClients, settings: Settings | None = None) -> "VectorialRAG":
        settings = settings or load_config()
        cfg = settings.qdrant
        client = AsyncQdrantClient(host=cfg.host, port=cfg.port)

        # Idempotent: only create the collection the first time the app runs.
        if not await client.collection_exists(cfg.collection):
            await client.create_collection(
                collection_name=cfg.collection,
                vectors_config=VectorParams(size=cfg.vector_size, distance=Distance[cfg.distance.upper()]),
            )
        return cls(client, clients, settings)

    async def ingest(self, content: str) -> None:
        vector = (await embed(self._clients.embeddings, [content]))[0]
        await self._client.upsert(
            collection_name=self._settings.qdrant.collection,
            points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": content})],
        )

    async def query(self, question: str, top_k: int = 5) -> list[dict]:
        vector = (await embed(self._clients.embeddings, [question]))[0]
        results = await self._client.query_points(
            collection_name=self._settings.qdrant.collection,
            query=vector,
            limit=top_k,
        )
        return [{"text": p.payload["text"], "score": p.score} for p in results.points]