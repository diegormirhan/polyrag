# ADR-0001: Estratégia de desempate na zona cinzenta do roteador semântico

**Status:** Proposta (em avaliação — nenhuma opção foi escolhida ainda)
**Data:** 2026-08-18
**Decisores:** Diego

## Contexto

O roteador semântico do PolyRAG (seção 5 do `CLAUDE.md`, "O ROTEADOR SEMÂNTICO") decide em 3
estágios, do mais barato ao mais caro:

1. Heurística (só na ingestão)
2. Cosseno + margem contra as âncoras do `config.yaml` (`tau_high = 0.62`, `tau_low = 0.35`,
   `delta_margin = 0.08`)
3. **LLM-as-judge**, acionado só na zona cinzenta (`tau_low ≤ score_top1 < tau_high` ou
   `margem < delta_margin`)

O estágio 3 é hoje a única etapa do roteamento que depende de um LLM. Isso destoa da tese central
do projeto (seção 0.1 do `CLAUDE.md`): "determinismo onde importa, IA onde faz diferença", e do
próprio glossário que classifica o LLM-judge como "custo alto, uso raro". A discussão que motivou
este ADR nasceu de um caso de estudo (`CASO-GRAPHIFY-DETERMINISMO.md`): o projeto Graphify alega
"determinismo" atribuindo o mérito à ausência de vetor, quando na verdade a origem real de
não-determinismo em qualquer pipeline de RAG é **qualquer etapa que dependa de um LLM com
amostragem** — o que expôs que o próprio LLM-as-judge do PolyRAG é, hoje, o único ponto do
roteamento onde essa fragilidade existe.

**Pergunta a responder:** existe uma alternativa que resolva a zona cinzenta sem LLM, mantendo (ou
melhorando) a acurácia de roteamento, sem comprometer o custo de manutenção nem o "cold start" do
projeto (zero dados rotulados no dia 1)?

## Decisão

Ainda não tomada. Este documento registra as opções em avaliação e os critérios de escolha.
Atualizar o `Status` para `Aceita` quando uma opção for selecionada, e mover as descartadas para
`Consequências` com o motivo da rejeição.

## Opções Consideradas

### Opção A — Manter LLM-as-judge (baseline atual)

| Dimensão | Avaliação |
|---|---|
| Complexidade de implementação | Baixa (já existe) |
| Custo por decisão | Alto (1 chamada de LLM por caso ambíguo) |
| Dados de treino necessários | Nenhum |
| Determinismo | Baixo (amostragem do LLM) |
| Auditabilidade | Média (dá para logar o prompt e a resposta, mas o "porquê" é uma caixa-preta) |
| Alinhamento com a tese do projeto | Fraco — é a exceção que a seção 0.1 do `CLAUDE.md` tenta minimizar |

**Prós:** funciona no dia 1, sem dataset; lida bem com perguntas genuinamente novas/fora da
distribuição, porque o LLM tem raciocínio semântico geral.
**Contras:** é o único ponto não-determinístico do roteamento; custo e latência variáveis; é
exatamente o tipo de alegação frágil que o `CASO-GRAPHIFY-DETERMINISMO.md` critica em outro
projeto — manter isso sem mitigação enfraquece o pitch de determinismo do PolyRAG.

### Opção B — Regressão logística multinomial (softmax regression)

Classificador linear treinado sobre features derivadas dos próprios scores do roteador
(`score_top1`, `score_top2`, `score_top3`, `margem`, entropia da distribuição de scores, sinais da
heurística do Dia 1). Decide por `argmax` da probabilidade prevista.

| Dimensão | Avaliação |
|---|---|
| Complexidade de implementação | Média (treino offline + inferência simples) |
| Custo por decisão | ~0 (produto escalar + softmax, sem LLM) |
| Dados de treino necessários | Sim — golden set rotulado (pergunta/chunk → rota correta) |
| Determinismo | Total (mesma entrada → mesma saída, sem amostragem) |
| Auditabilidade | Alta — pesos são interpretáveis, testável com `pytest` puro, igual ao resto da matemática do roteador |
| Alinhamento com a tese do projeto | Forte — elimina a última chamada de LLM do estágio de roteamento |

**Prós:** reforça a tese "100% da decisão de rota é matemática" citada na seção 0.1 do
`CLAUDE.md`; custo e latência previsíveis; reusa o golden set já planejado para o Dia 11
(avaliação offline).
**Contras:** cold start — precisa de dados rotulados antes de existir qualquer benefício; risco de
overfitting num golden set pequeno (50–100 exemplos) sem regularização/validação cruzada; não
generaliza para perguntas cujo padrão de scores nunca apareceu no treino (o LLM-judge, em teoria,
lida melhor com o inédito).

### Opção C — Cascata: regressão logística com fallback para LLM-as-judge

Usa a Opção B como primeiro estágio da zona cinzenta; só cai no LLM-as-judge quando a própria
regressão logística também está insegura (probabilidade máxima abaixo de um limiar configurável,
ex. `tau_judge_fallback`).

| Dimensão | Avaliação |
|---|---|
| Complexidade de implementação | Média-alta (dois estágios, mais um threshold para calibrar) |
| Custo por decisão | Baixo em média (LLM só nos casos residuais mais difíceis) |
| Dados de treino necessários | Sim, mas o fallback cobre a lacuna enquanto o dataset é pequeno |
| Determinismo | Alto na maioria dos casos; preserva rede de segurança para o resto |
| Auditabilidade | Alta — span pode registrar `router.decision_stage = "logreg"` ou `"llm_judge_fallback"` |
| Alinhamento com a tese do projeto | Forte, e mais honesto: admite explicitamente onde a IA ainda entra, em vez de fingir 100% de cobertura determinística |

**Prós:** reduz uso de LLM sem eliminar a rede de segurança para casos fora da distribuição;
migração incremental (pode nascer com o fallback dominante e ir invertendo a proporção conforme o
dataset cresce); span já registra a proporção `logreg vs. llm_judge_fallback` como métrica de
maturidade do roteador ao longo do tempo.
**Contras:** mais peças móveis que as opções A e B isoladas; precisa calibrar dois limiares
(`tau_high`/`delta_margin` da entrada e `tau_judge_fallback` da saída da regressão) em vez de um.

### Opção D — Fórmula de score ponderado, sem treino (heurística manual)

Combinação linear manual dos mesmos scores (ex. `score_top1 * w1 + margem * w2 + heuristica * w3`)
com pesos escolhidos por tentativa e erro / conhecimento de domínio, sem ajuste estatístico.

| Dimensão | Avaliação |
|---|---|
| Complexidade de implementação | Baixa |
| Custo por decisão | ~0 |
| Dados de treino necessários | Nenhum (mas precisa de exemplos para *validar* os pesos escolhidos) |
| Determinismo | Total |
| Auditabilidade | Alta, mas os pesos não têm justificativa estatística — é uma heurística disfarçada de fórmula |
| Alinhamento com a tese do projeto | Parcial — é determinístico, mas troca "LLM opaco" por "pesos arbitrários", sem o rigor de um modelo ajustado a dados |

**Prós:** zero dependência de dataset rotulado; mais simples que B/C.
**Contras:** sem calibração estatística, tende a herdar os mesmos vieses do threshold atual sem
resolver o problema original (é basicamente uma variação do estágio 2, não uma solução nova para o
estágio 3); mantida aqui só como piso de comparação para justificar se B/C realmente compensam a
complexidade extra.

## Trade-off Analysis

O eixo principal é **determinismo/custo vs. cold start**. As opções B e C exigem o golden set que
o roadmap já planeja para o Dia 11 (`CLAUDE.md`, seção 9) — a decisão real, portanto, também é
**quando** essa troca acontece: nada impede começar o projeto com a Opção A (LLM-as-judge) e
migrar para B ou C assim que o golden set existir, documentando isso como uma migração planejada
em vez de indecisão. A Opção C parece o melhor compromisso entre honestidade da tese de
determinismo e robustez a casos fora da distribuição, mas adiciona um segundo limiar para calibrar
e um estágio a mais para testar. A Opção D serve principalmente como baseline de descarte — provável
que não valha a complexidade frente a B/C uma vez que haja dados.

## Consequências

- Ainda não aplicável — nenhuma opção foi aceita. Atualizar esta seção quando a decisão for
  tomada, com o que passa a ser mais fácil (ex. testes unitários determinísticos do estágio 3) e
  mais difícil (ex. necessidade de manter/retreinar um modelo) na opção escolhida.

## Action Items

1. [ ] Definir o schema mínimo do golden set (pergunta/chunk, rota correta, scores do roteador) —
   pode ser o mesmo dataset do Dia 11, adiantado.
2. [ ] Rodar a Opção A (LLM-as-judge) em produção/dev por um tempo só para **coletar** os casos de
   zona cinzenta reais (span já registra `router.decision_stage`) — isso vira o dataset de treino
   sem esforço extra de rotulagem manual do zero.
3. [ ] Prototipar a Opção B offline com esse dataset coletado e medir `router_accuracy` vs. o
   LLM-judge no mesmo conjunto de casos.
4. [ ] Se B tiver acurácia comparável ao LLM-judge, decidir entre B (puro) e C (cascata) com base
   no apetite a risco para casos fora da distribuição.
5. [ ] Atualizar `Status` para `Aceita` e preencher `Consequências` com a escolha final.
