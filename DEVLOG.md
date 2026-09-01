# PolyRAG — Dev Log

> Registro cronológico das decisões, bugs reais e descobertas durante o desenvolvimento — não é
> changelog de features, é o "porquê" por trás de cada decisão, pra reconstruir o raciocínio depois
> (entrevista, retomada do projeto, ou só memória).

---

## Dia 1 — Ingestão (2026-08-24)

### O que foi construído
`core/config.py` (loader Pydantic do `config.yaml`) · `core/llama_client.py` (wrapper async sobre
`AsyncOpenAI` pros 3 `llama-server`) · `ingest/watcher.py` (poll em `data_drop/`, classifica por
extensão) · `ingest/loaders.py` (texto/tabela/documento — pdf/docx/pptx) · `ingest/ocr.py`
(GLM-OCR) · `ingest/chunking.py` (chunking semântico com preservação de tabela).

### Bugs reais encontrados e corrigidos (não hipotéticos — todos apareceram rodando código de verdade)

1. **Config com typos self-inflicted:** `port: str` em vez de `int` na config de embeddings,
   `collections` no lugar de `collection` (não batia com a chave do YAML — quebraria o
   `load_config()` com "field required"), import não usado (`from asyncio import graph`).
2. **Caminho do binário do llama.cpp divergiu do planejado:** `config.yaml` apontava pra
   `runtime/llama/`, mas o binário real ficou em `bin/llama-server/`. Atualizado o config pro
   caminho real em vez de mover pastas.
3. **`pd.read_html()` não aceita mais string literal de HTML** (mudança de comportamento de versão
   recente do pandas) — precisa envolver em `io.StringIO()`, senão ele tenta tratar a string como
   caminho de arquivo/URL e explode com `FileNotFoundError`.
4. **Lógica invertida em `Watcher.poll_once()`:** uma versão simplificada por engano acabou com
   `if kind is None: ready.append(...)` — adicionando os arquivos **não reconhecidos** em vez dos
   reconhecidos. Pego na revisão antes de rodar.
5. **`_split_table_blocks` comparava `match.start() > 0` em vez de `> pos`** — inofensivo hoje
   (mascarado por um `.strip()` mais adiante), mas lógica errada mesmo assim.
6. **Mojibake nos acentos ao rodar script direto no terminal do Windows** — não era bug no
   `loaders.py` (confirmado lendo com `PYTHONUTF8=1`), era só o console não configurado pra UTF-8.
   Fica documentado pra não perder tempo com isso de novo.

### Descoberta: GLM-OCR não obedece instrução de formato — e a causa é treinamento, não bug

O `ocr_prompt` pedia "reproduza tabelas em Markdown", o modelo devolvia **HTML** mesmo assim.
Insistir com "nunca em HTML" não mudou nada. Fomos na doc oficial
(`llama.cpp/docs/multimodal.md`) e na discussão do PR que adicionou o GLM-OCR
(ggml-org/llama.cpp#19677): *"como todo modelo de OCR, deve ser usado com prompts específicos"* —
e o exemplo deles manda literalmente a palavra `"OCR"`. Trocamos o prompt de uma instrução longa
pra só `"OCR"` — resultado: **extraiu o recibo inteiro** (antes só pegava a tabela) e a segunda
imagem saiu em Markdown puro, sem precisar nem do pós-processamento HTML→Markdown que tínhamos
construído antes. Lição: modelos de OCR especializados têm formato de prompt estreito aprendido no
fine-tuning — instrução elaborada pode confundir mais que ajudar, diferente de um LLM genérico.

### Descoberta: não-determinismo real no OCR, e por que isso importava especialmente aqui

Rodando a mesma imagem (`table.png`) duas vezes, um Id saiu `249` numa rodada e `248` (duplicado)
noutra — o mesmo arquivo, resultado diferente. Causa: `ocr.py` chamava `chat()` sem especificar
`temperature`, caindo no default de `0.2`. Para extração de texto não existe benefício de
"criatividade" — sampling só arrisca alucinar um dígito. Corrigido com `temperature=0` explícito.
Achado relevante porque bate direto na tese central do projeto (seção 0.1 do `CLAUDE.md`):
"determinismo onde importa" — e OCR é claramente um lugar onde importa, mesmo não estando na tabela
original de "etapas determinísticas" do CLAUDE.md (LLM de extração deveria se comportar o mais
determinístico possível, já que não há ambiguidade legítima em "que texto está na imagem").

### Evolução do chunking (3 iterações, guiadas por dado real, não por teoria)

1. **Sliding window puro** (Dia 1 original) — simples, funcionou pro caso trivial.
2. **+ Preservação de tabela** — motivada por um bug real: uma tabela Markdown foi cortada no meio
   entre dois chunks no teste do `receipt.png`. Detecta blocos `|...|` consecutivos via regex e
   trata como unidade atômica, nunca fatiada.
3. **+ Chunking semântico** (substituiu sliding window por completo, decisão do usuário) — corta
   por mudança de assunto (cosseno entre embeddings de frase) em vez de contagem cega de
   caracteres. Passou por 3 rodadas de calibração:
   - Threshold `0.75` inicial → fragmentou um recibo em chunks de **3 caracteres** (`"NO."`) porque
     o sentence-splitter trata ponto de abreviação como fim de frase, e comparar frase-a-frase é
     ruidoso.
   - Adicionado piso de tamanho mínimo (`min_chunk_chars`) — força continuar no mesmo grupo abaixo
     de um tamanho, mesmo com cosseno baixo. Resolveu os fragmentos inúteis.
   - Trocada a comparação de "frase anterior" por "centróide (média) do grupo acumulado até agora"
     — reduz ruído de comparação par-a-par. Validado com um arquivo de teste propositalmente criado
     com **2 assuntos bem diferentes** (receita de pão vs. buracos negros): confirmado que o corte
     cai exatamente na fronteira real entre os dois parágrafos, não em lugar aleatório.
   - Limitação conhecida e aceita por ora: a primeira comparação de um grupo recém-aberto ainda é
     ruidosa (centróide de 1 vetor = o próprio vetor) — fica pra revisitar só se aparecer problema
     real de recall depois (Dia 11).

### Backlog de técnicas avançadas — deliberadamente adiado

Numa única conversa surgiram 8 técnicas adicionais de RAG (clarification loop, multi-interpretação/
contradição, "lost in the middle", recursive/semantic chunking adicional, parent-child chunking,
re-ranker, late chunking, metadata rica). Documentadas na íntegra no `CLAUDE.md` (seção de backlog),
cada uma mapeada pro dia certo do roadmap, com nota de confiança sobre o ROI. Só a preservação de
tabela e o chunking semântico foram puxados pra agora, porque tinham bug real motivando.
**Padrão a repetir:** trazer ideia boa não é sinal pra implementar na hora — é sinal pra registrar e
avaliar contra o princípio de simplicidade.

### Avaliação crítica de portfólio (autoavaliação puxada pelo usuário)

Pontos fortes reais: math visível e testável (não escondida em lib), narrativa de determinismo
defensável, 100% local em hardware não-CUDA (consistente com outros projetos pessoais), processo de
debugging documentado (isso — não a arquitetura em si — é o que mais vende em entrevista).

Riscos nomeados sem suavizar: cronograma de 2 semanas é otimista (Dia 1 "fácil" já tomou uma sessão
inteira); scope creep é risco demonstrado nesta mesma sessão (8 técnicas propostas de uma vez); RAG 3
(GraphRAG/HippoRAG 2 via LLM local de 14B) é o componente de maior risco técnico do projeto inteiro;
avaliação (Dia 11) ainda é 100% meta, não medição — é o que prova a tese central, deve ser protegida
antes de qualquer polish extra se o tempo apertar.

**Atualização de escopo (mesma conversa):** frontend (Dia 5/10) será construído em modo
"vibecode" (iteração rápida assistida por IA, menos deliberação manual) — reduz o risco de tempo
que eu tinha levantado pro frontend. Ainda vale proteger o Dia 11 (avaliação) como prioridade caso o
tempo aperte de qualquer forma, já que vibecode ajuda a *construir* rápido mas não substitui
*medir* se o sistema funciona.

---

## Dia 2 — Roteador semântico (2026-08-31)

### O que foi construído
`core/vectors.py` (`normalize`/`dot` compartilhados) · `pipeline/tabularity.py` (Estágio 1) ·
`pipeline/router.py` (3 estágios) · `tests/test_router_math.py` (5 testes puros, sem LLM).

### Decisões
- **`_TABLE_BLOCK` virou público (`TABLE_BLOCK`)** e é importado por `tabularity.py` em vez de
  redefinido — a definição de "o que é uma tabela Markdown" tem uma fonte só.
- **`normalize`/`dot` saíram do `chunking.py`** para `core/vectors.py` quando o roteador passou a
  precisar da mesma matemática. Mesma regra: nada de cosseno duplicado em dois arquivos.
- **Tentativa de quebrar `route()` em métodos por estágio foi revertida.** Os métodos passaram a
  precisar de 7 parâmetros cada (variável local não é compartilhada entre métodos só por estarem na
  mesma classe), e ler "o que acontece na zona cinzenta" exigia pular entre 3 funções. Função única
  com `return` antecipado por estágio ficou mais legível. **Dividir função só compensa quando o
  pedaço é reusado, esconde complexidade real, ou não cabe na cabeça de uma vez.**

### Troca de modelos (Qwen3-14B → Qwen3-8B + Prometheus 2 dedicado)
O papel de juiz saiu do Qwen3 e virou modelo dedicado (**Prometheus 2 7B**, especializado em
avaliação, porta 8083). Pra abrir VRAM, o Qwen3-14B (9.5GB) virou **Qwen3-8B** (5.03GB) — mesma
família de propósito, evitando as "manias de prompt" de trocar de família (lição do GLM-OCR).
Dado real que motivou: o Windows + apps de fundo já consomem ~4GB de VRAM antes de qualquer modelo
subir, então o orçamento efetivo é ~12GB, não os 16GB nominais.

### Melhorias no LLM-as-judge, vindas de fonte externa
Consulta ao guia da DeepEval sobre LLM-as-a-judge trouxe duas mudanças concretas:
1. **`evaluation_steps` explícitos** em vez de critério vago — o prompt do juiz passou a listar os
   passos de raciocínio ("identifique o tipo de conteúdo → compare com cada rota → escolha").
2. **Checagem de viés de posição:** juízes LLM (e quanto menores, pior) tendem a favorecer a opção
   listada primeiro. `_llm_judge` agora pergunta **duas vezes com a ordem invertida** e só aceita se
   as duas concordarem; divergência = viés detectado → cai no vencedor determinístico do Estágio 2.

### Validação (3 estágios exercitados de verdade)
| Caso | Estágio | Rota | top1 | margem |
|---|---|---|---|---|
| Tabela Markdown | heuristic | relational | 1.000 | 1.000 |
| Receita de pão | llm_judge | vectorial | 0.382 | 0.098 |
| Regra de dependência | embedding | graph | 0.671 | 0.321 |
| Pergunta agregada SQL | embedding | relational | 0.674 | 0.359 |
| Ambíguo (proposital) | llm_judge | vectorial | 0.501 | **0.012** |

O caso ambíguo caiu com margem 0.012 — zona cinzenta de manual. Nos 3 casos que chegaram ao juiz,
as duas rodadas (ordem normal e invertida) concordaram: sem viés de posição detectado.

**Achado para calibrar depois:** a receita de pão — exemplo mais óbvio possível de "vectorial" —
teve `top1=0.382`, baixo demais. Suspeita: as `utterances` do `config.yaml` são todas **perguntas**
("resuma o documento sobre..."), mas na ingestão o roteador recebe **conteúdo declarativo**. Esse
desalinhamento de registro pode estar empurrando chunks demais pra zona cinzenta (caminho caro).
Candidato a Dia 11.

---

## Dia 3 — As 3 bases (2026-08-31)

### O que foi construído
`rags/base.py` (interface) · `rags/relational.py` (RAG 1) · `rags/vectorial.py` (RAG 2) ·
`rags/graph.py` (RAG 3) · testes puros de `validate_read_only` (10) e `parse_triples` (5) ·
3 testes de integração. Infra nova: Qdrant v1.19.0 nativo (`bin/qdrant/`), `qdrant-client`,
`networkx`, `scipy`.

### RAG 1 e RAG 2 — funcionaram de primeira
- **Relacional:** Qwen3-8B gerou `SELECT SUM(vendas) FROM vendas WHERE regiao = 'Sudeste'` e o
  resultado saiu **320** (= 120 + 200, os dois registros do Sudeste). Demonstra na prática a tese de
  determinismo: o número está certo porque o SQL está certo, sem LLM fazendo aritmética.
- **Vetorial:** ranking correto nas duas perguntas de teste, com separação clara entre relevante e
  irrelevante (0.643 vs 0.298).

### DESCOBERTA GRANDE: thinking mode ligado por padrão no Qwen3-8B
O teste do grafo travou (>5min sem terminar um único chunk). Investigando com `curl` direto no
`llama-server`, um pedido trivial ("responda só OK") devolveu um campo `reasoning_content` com ~90
tokens de raciocínio interno.

| | tempo | tokens |
|---|---|---|
| Thinking ON (padrão) | **3.06s** | ~90 de raciocínio + resposta |
| `enable_thinking: false` | **0.62s** | 3 |

**5x mais rápido** numa chamada trivial; em prompt few-shot longo (OpenIE) a diferença é o que
separava "roda em segundos" de "não termina em 5 minutos". O código nunca quebrou porque o
llama-server separa o raciocínio em `reasoning_content` e o nosso `chat()` lê só `content` — mas
pagávamos o custo de gerar tudo aquilo em toda chamada. Corrigido em `chat()` com
`extra_body={"chat_template_kwargs": {"enable_thinking": ...}}`, default `False` (tarefas
estruturadas não ganham nada com raciocínio; segue o mesmo padrão do `temperature`).

**Ironia útil:** eu tinha levantado thinking mode como argumento pra *adiar* a troca pelo
Qwen3.5-9B. O problema já estava no modelo atual — só ninguém tinha medido. Lição: suspeita de
performance vira fato com uma medição de 30 segundos.

### RAG 3 — dois bugs reais, encontrados só porque o teste era multi-hop de verdade
O teste foi desenhado com uma cadeia proposital de 3 saltos (Alfa → Beta → Motor X → Contrato Gama),
onde **nenhum chunk menciona as duas pontas** — assim, só o grafo poderia recuperar o último.

**Bug 1 — busca devolvia zero resultados.** O `openie_prompt` diz "extraia apenas relações
explicitamente declaradas; se não houver, responda `[]`". Uma **pergunta não declara fato nenhum** →
o modelo obedeceu corretamente e devolveu `[]` → nenhuma semente → nenhum resultado. Erro meu de
design: o mesmo prompt não serve chunk (declarativo) e pergunta (interrogativa). E divergia do
paper: **HippoRAG 2 usa NER na pergunta**, não extração de triplas. Corrigido com um `ner_prompt`
separado.

**Bug 2 — grafo fragmentado em ilhas desconectadas.** As entidades saíram com nomes inconsistentes
entre chunks:
```
"O Fornecedor Alfa"  /  "componentes eletronicos para a Fabrica Beta"  (objeto virou frase inteira)
"A Fabrica Beta"     /  "Motor X"  →  virou "item_principal" no chunk seguinte (LLM renomeou)
"Contrato_Gama"      (underscore, inconsistente com o resto)
```
A cadeia virou 3 ilhas sem nenhuma aresta entre elas — multi-hop impossível. Duas correções:
1. **`normalize_entity()`** — remove artigo inicial, troca `_` por espaço, `strip`. Pura e testável.
2. **Few-shot no `openie_prompt`** — mostrar um exemplo concreto do formato desejado funcionou muito
   melhor que descrever regras em texto. **Mesma lição do GLM-OCR:** instrução abstrata é fraca; o
   modelo precisa ver o padrão.

### Resultado final do multi-hop (o que valida o RAG 3 existir)
```
score=0.1921 | chunk 1: Fornecedor Alfa → Fabrica Beta     (1 salto)
score=0.1096 | chunk 2: Fabrica Beta → Motor X             (2 saltos)
score=0.0309 | chunk 3: Motor X → Contrato Gama            (3 saltos)
score=0.0000 | chunk 4: política de férias (desconectado)
```
Score decai com a distância no grafo, e o chunk irrelevante ficou em zero absoluto (nenhum caminho).
O chunk 3 **não contém "Fornecedor Alfa"** — busca vetorial nunca o traria; só a propagação no grafo.

**Achado para calibrar depois:** as sementes saíram `['Fabrica Beta', 'Fornecedor Alfa']`, mas a
pergunta só menciona *Fornecedor Alfa*. "Fabrica Beta" casou por cosseno ≥ 0.6 contra a entidade da
pergunta — dois nomes próprios diferentes acima do limiar. Aqui ajudou, mas sugere que
`entity_match_threshold: 0.6` está permissivo demais (nomes próprios em português têm estrutura
parecida entre si). Candidato a Dia 11.

---

## Como usar este arquivo

Uma entrada nova por dia do roadmap (ou por sessão relevante), sempre estruturada como: o que foi
construído → bugs reais encontrados (não hipotéticos) → descobertas que mudaram uma decisão →
o que ficou pra depois e por quê. Não documentar aqui o que já está óbvio lendo o código — só o
raciocínio que não sobrevive só olhando o `git log`.
