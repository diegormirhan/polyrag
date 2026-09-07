# Demo corpus

Six files about one fictional company — Meridiano Logística, a Brazilian freight
and warehousing operator. One company rather than three unrelated samples, so the
demo tells a story instead of showing three toys.

The documents are in Portuguese because answers follow the language of the
question, and a demo where the interface is English and the answer is Portuguese
is what this system actually does.

| File | Route it should reach | Why |
|---|---|---|
| `vendas_2026_q1.csv` | relational | 40 rows of revenue by month, region and product |
| `sobre_a_meridiano.md` | vectorial | company history — narrative, no rules, no tables |
| `manual_atendimento.md` | vectorial | customer-service guidance, written as prose |
| `politicas_compras.md` | graph | approval thresholds, contracts, supplier certification |
| `normas_dados.md` | graph | data-retention rules and audit obligations |
| `dependencias_sistemas.md` | graph | which system depends on which |

## Loading it

```bash
uv run python scripts/load_demo.py --reset
```

`--reset` empties the three stores first. Without it the demo inherits whatever
the last run left behind, and stale chunks are exactly what makes a
demonstration confusing.

With the backend running, the hot folder ingests within seconds. Without it:

```bash
uv run python -u tests/test_ingest_integration.py
```

Where the 24 chunks land, from `GET /api/v1/corpus`:

| File | Stored in |
|---|---|
| `vendas_2026_q1.csv` | relational 1 |
| `sobre_a_meridiano.md` | vectorial 5 |
| `manual_atendimento.md` | vectorial 5 |
| `politicas_compras.md` | graph 4, vectorial 1 |
| `normas_dados.md` | graph 3, vectorial 1 |
| `dependencias_sistemas.md` | graph 3, vectorial 1 |

Three chunks sit in the vector store despite being about rules. That is the
ingestion fallback, not a routing miss: the router judges meaning, the graph
needs an extractable relation, and text that yields no triple is kept as free
text rather than dropped. Before that fallback existed those chunks were in no
store at all.

## The script

### Relational — arithmetic must be exact

```
Qual foi a receita total da região Sudeste?
```

Expect **4.988.300**. The number comes from a generated `SUM`, not from a model
adding figures in prose. Watch the panel: `rag.relational.generate_sql` carries
the query it wrote, which is the audit trail for the answer.

Other verified figures: total revenue 12.950.950 · March 3.293.750 · Nordeste
2.500.700 · Armazenagem 1.882 units.

### Vectorial — meaning, with nothing to match on literally

```
Em que ano a Meridiano Logística foi fundada?
```

Answer: **2011**, in a rented room in Belém. The figure sits inside a paragraph
of prose with no table and no heading naming it, so it can only be found by
meaning.

Questions may be asked in English against this Portuguese corpus. Routing and
retrieval work across the two because BGE-M3 embeds both into one space, and the
answer follows the language of the question while the sources stay as written.

### Graph — multi-hop, and the reason this store exists

```
Uma compra de oitenta mil reais do Fornecedor Vega precisa de qual aprovação?
```

Answer: **Diretoria Financeira**. Three hops — Fornecedor Vega → Contrato Marco
2026 → Política PC-04 → the fifty-thousand threshold. No single chunk names the
supplier and the approver together.

```
Os pedidos processados pelo Sistema Atlas seguem qual política de aprovação?
```

Answer: **Política PC-09**, dual approval by the Regional Manager and the
Operations Board. This is the one to show: `Sistema Atlas` lives in
`dependencias_sistemas.md` and `Política PC-09` in `politicas_compras.md`. The
two files were ingested separately and never mention each other. The graph joins
them through the entity `pedidos`, which both documents happen to talk about.

```
O Fornecedor Vega ainda pode ser homologado?
```

Answer: **no** — it lost its ISO 9001 in February, and homologation depends on a
valid certificate. A judgement assembled from two separate statements rather
than a fact retrieved from one.

### Cache — the same question twice

Ask any of the above a second time. The panel says
`Answered from the semantic cache. No routing, no retrieval, no model call.`
and it returns in roughly **8 ms** against seconds for the full path. A
paraphrase hits it too — the match is on the embedding, not the string.

## What does not work, and why it is here

```
O que motivou a criação do Sistema Atlas?
```

The answer is in the company history: two distribution centres kept separate
spreadsheets after the 2018 Recife expansion, and monthly reconciliation ate a
week of manual work. It is a narrative question, and it goes to the **graph**.

The cause is visible in the router's scoreboard. Each store contributes the
section headings it holds, and the graph holds a section literally called
`Sistema Atlas`. The question names it, so the graph wins on evidence that is
real but beside the point: the graph knows the entity, it just does not hold the
story. Entity presence is not the same as answer presence, and nothing in the
current design distinguishes them.

This section used to document a different failure, which the measurements fixed:

```
Se o Sistema Atlas ficar indisponível, qual política de aprovação é afetada?
```

That one used to answer that the context established no relation. Named-entity
extraction was pulling generic concepts out of the question (`aprovação prévia`,
`dupla aprovação`) and seeding PageRank with them, which ranked PC-04 above
PC-09. Seeding from the question's own embedding and fusing the PageRank order
with direct similarity fixed it: it now answers **Política PC-09**, correctly.

Neither the old failure nor the new one was patched to make the demo look
better. A corpus exists to find these, and the golden set is what decides
whether they are worth fixing.
