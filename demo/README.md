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

Measured on this corpus: **23 of 24 chunks routed as expected.** The one miss is
a paragraph of `manual_atendimento.md` about handling complaints, which reads
like a rule and goes to the graph. Worth showing rather than hiding — it is what
a threshold-based decision looks like near its boundary.

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
O que motivou a criação do Sistema Atlas?
```

The answer is in the company history: two distribution centres kept separate
spreadsheets after the 2018 Recife expansion, and monthly reconciliation ate a
week of manual work. The question shares almost no words with that passage, so
keyword search would miss it.

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
Se o Sistema Atlas ficar indisponível, qual política de aprovação é afetada?
```

This one **fails**, and the failure is worth understanding because it is not
where you would expect.

Retrieval is fine: the PC-09 chunk comes back ranked second and the Atlas chunk
third, so the model receives both halves of the chain. It still answers that the
context establishes no relation. The gap is in generation — the answer prompt
caps the response at three sentences with no reasoning steps, and a 4B model
needs room to make the inference that the terser question above does not
require.

Two things visible in the same trace, both already queued for measurement:

- The seeds include `aprovação prévia`, `dupla aprovação` and `aprovação formal`.
  Named-entity extraction pulled generic concepts out of the question and
  `entity_match_threshold: 0.6` matched all of them, which is what ranks PC-04
  above PC-09.
- Retrieval scores cluster tightly (0.2028, 0.1009, 0.0387), so no fixed floor
  separates useful from useless here.

Neither is patched to make the demo look better. A corpus exists to find these,
and the golden set is what decides whether they are worth fixing.
