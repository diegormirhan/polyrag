import asyncio

from app.core.llama_client import LlamaClients
from app.rags.graph import GraphRAG

# Deliberate 3-hop chain: Alfa -> Beta -> Motor X -> Contrato Gama.
# No single chunk mentions both "Fornecedor Alfa" and "Contrato Gama", so a plain
# vector search on the question would never surface the last one — only walking
# the graph does.
CHUNKS = [
    "O Fornecedor Alfa fornece componentes eletronicos para a Fabrica Beta.",
    "A Fabrica Beta produz o Motor X usando os componentes recebidos.",
    "O Motor X e o item principal entregue no Contrato Gama.",
    "A politica de ferias da empresa exige aviso previo de trinta dias.",
]

QUESTION = "Se o Fornecedor Alfa parar de entregar, o que e afetado?"


async def main() -> None:
    clients = LlamaClients()
    rag = GraphRAG.load(clients)

    print("===== INGESTAO (OpenIE -> grafo) =====", flush=True)
    for chunk in CHUNKS:
        await rag.ingest(chunk)
        print(f"  ingerido: {chunk[:60]}...", flush=True)

    graph = rag._graph
    print("\ntriplas no grafo:")
    for u, v, data in graph.edges(data=True):
        if data.get("relation") != "mentioned_in":
            print(f"  ({u}) --{data['relation']}--> ({v})")
    entities = [n for n, d in graph.nodes(data=True) if d.get("kind") == "entity"]
    print(
        f"\ngrafo: {graph.number_of_nodes()} nos ({len(entities)} entidades), "
        f"{graph.number_of_edges()} arestas"
    )

    print("\n===== BUSCA (PPR multi-hop) =====")
    print(f"pergunta: {QUESTION}")
    results = await rag.query(QUESTION, top_k=4)
    if not results:
        print("nenhuma semente casou — nenhum resultado")
        return

    print(f"sementes: {results[0]['seeds']}\n")
    for hit in results:
        print(f"  score={hit['score']:.4f} | {hit['text']}")


if __name__ == "__main__":
    asyncio.run(main())
