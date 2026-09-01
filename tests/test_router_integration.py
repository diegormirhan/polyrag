import asyncio

from app.core.llama_client import LlamaClients
from app.pipeline.router import Router

SAMPLES = [
    (
        "tabela markdown (deveria bater no Estágio 1 - heurística)",
        "| produto | vendas |\n|---|---|\n| Notebook | 120 |\n| Mouse | 80 |",
    ),
    (
        "narrativa não-estruturada (deveria bater no Estágio 2 - vectorial)",
        "Para fazer pão caseiro, primeiro misture farinha, água, fermento e sal numa tigela "
        "grande. Sove a massa por dez minutos até ficar lisa e elástica.",
    ),
    (
        "regra de negócio/dependência (deveria bater no Estágio 2 - graph)",
        "Se o fornecedor principal atrasar a entrega, os contratos B e C ficam automaticamente "
        "suspensos até nova aprovação da diretoria.",
    ),
    (
        "pergunta agregada tipo SQL (deveria bater no Estágio 2 - relational)",
        "Qual foi o total de vendas da região Sudeste no último trimestre?",
    ),
    (
        "ambíguo de propósito (candidato a zona cinzenta - Estágio 3)",
        "O relatório trimestral menciona que uma mudança no fornecedor principal pode "
        "impactar outros processos internos da empresa, mas não detalha quais.",
    ),
    (
        "ambíguo de propósito #2 (candidato a zona cinzenta - Estágio 3)",
        "A política interna descreve, em texto corrido, como diferentes departamentos "
        "dependem uns dos outros para aprovar uma compra.",
    ),
]


async def main() -> None:
    clients = LlamaClients()
    router = await Router.create(clients)

    for label, text in SAMPLES:
        decision = await router.route(text)
        print(f"\n=== {label} ===")
        print(f"texto: {text[:80]}...")
        print(
            f"route={decision.route} | stage={decision.decision_stage} | "
            f"top1={decision.score_top1:.3f} | top2={decision.score_top2:.3f} | margin={decision.margin:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
