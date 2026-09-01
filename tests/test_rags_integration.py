import asyncio

import pandas as pd

from app.core.llama_client import LlamaClients
from app.rags.relational import RelationalRAG
from app.rags.vectorial import VectorialRAG

CHUNKS = [
    "Para fazer pão caseiro, misture farinha, água, fermento e sal numa tigela grande.",
    "Buracos negros são regiões do espaço onde a gravidade é tão intensa que nem a luz escapa.",
    "O horizonte de eventos marca o limite a partir do qual nada mais consegue retornar.",
    "A política de compras exige aprovação da diretoria para valores acima de dez mil reais.",
]

QUESTIONS = [
    "o que acontece perto de um buraco negro?",
    "como se faz pão?",
]


async def check_relational(clients: LlamaClients) -> None:
    print("\n===== RAG 1 (relacional / SQLite) =====")
    rag = RelationalRAG(clients)

    df = pd.DataFrame(
        {
            "produto": ["Notebook", "Mouse", "Teclado", "Monitor"],
            "regiao": ["Sudeste", "Sul", "Nordeste", "Sudeste"],
            "vendas": [120, 80, 45, 200],
        }
    )
    await rag.ingest((df, "vendas"))
    print(f"schema:\n{rag.schema()}")

    question = "Qual o total de vendas da regiao Sudeste?"
    sql = await rag.generate_sql(question)
    print(f"\npergunta: {question}")
    print(f"SQL gerado: {sql}")
    print(f"resultado: {await rag.query(question)}")


async def check_vectorial(clients: LlamaClients) -> None:
    print("\n===== RAG 2 (vetorial / Qdrant) =====")
    rag = await VectorialRAG.create(clients)

    for chunk in CHUNKS:
        await rag.ingest(chunk)
    print(f"{len(CHUNKS)} chunks ingeridos")

    for question in QUESTIONS:
        print(f"\npergunta: {question}")
        for hit in await rag.query(question, top_k=2):
            print(f"  score={hit['score']:.3f} | {hit['text'][:70]}")


async def main() -> None:
    clients = LlamaClients()
    await check_relational(clients)
    await check_vectorial(clients)


if __name__ == "__main__":
    asyncio.run(main())
