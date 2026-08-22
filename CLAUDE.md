# PolyRAG — Plano de Implementação e Roadmap

> **Framework open-source de Unified/Federated Multi-Modal RAG com Roteamento Semântico e CAG.**
> Windows nativo · AMD Vulkan (RX 9060 XT 16GB) · sem PyTorch · sem vLLM · sem Ollama · sem Docker/WSL.

---

## 0. O projeto em uma frase (e o pitch de 30 segundos)

**Em uma frase:** um sistema onde você joga qualquer arquivo numa pasta (texto, planilha, imagem) e ele aprende sozinho onde guardar cada informação — depois responde perguntas escolhendo *matematicamente* qual dos 3 "cérebros" consultar.

**O pitch (para explicar a qualquer pessoa, até fora da tech):**

> "Imagine uma empresa com 3 tipos de funcionário: um **contador** (entende tabelas e números), um **bibliotecário** (entende textos e documentos) e um **detetive** (entende regras e conexões entre as coisas). Quando chega um documento novo ou uma pergunta, um **recepcionista matemático** decide, em milissegundos, qual dos 3 é o certo para o trabalho — e ele mostra o 'raciocínio' dele num painel, com as notas de confiança de cada decisão. Eu construí esse prédio inteiro do zero: o recepcionista usa geometria de vetores, o detetive usa o mesmo algoritmo que o Google inventou para ranquear páginas (PageRank), e tudo roda localmente na minha GPU AMD, sem nuvem."

### 0.0.1 Princípio de desenvolvimento — simplicidade primeiro, sem redundância

Este projeto é construído **incrementalmente, começando pelo mais simples possível**. Regras:

- Nunca adicionar robustez, tratamento de erro, ou lógica de borda para um cenário que ainda
  não aconteceu de verdade. Se surgir um bug concreto durante o desenvolvimento, resolve-se
  *naquele momento*, com o caso real guiando a solução — não antes, por precaução.
- Evitar código redundante: se duas partes fazem a mesma verificação ou guardam o mesmo estado
  de formas diferentes, simplificar para uma única fonte de verdade.
- Cada módulo deve nascer do jeito mais enxuto que resolve o problema do dia. Complexidade é
  adicionada de forma incremental, conforme o desenvolvimento avança e exige — nunca antecipada.
- Isso vale para todo o projeto, não só um módulo específico.

**Por que isso impressiona tanto Engenharia de Software quanto Data Science:**
- *Engenharia:* API versionada, WebSockets, observabilidade real (OpenTelemetry), config-driven, CI, arquitetura modular de verdade.
- *Data Science:* a matemática não está escondida numa biblioteca — está implementada e visível: similaridade de cosseno, margens de confiança, PageRank personalizado, quantização.

---

## 0.1 POSICIONAMENTO — o que é diferente (e o problema que isso resolve)

### O problema real que o projeto ataca: decisões NÃO-determinísticas

A maior dor das empresas com soluções de IA/RAG de mercado (agentes, function-calling, frameworks) é: **o comportamento é imprevisível**. A mesma pergunta pode:
- rotear para bases de dados **diferentes** a cada vez,
- gerar respostas **inconsistentes**,
- ter custo e latência **imprevisíveis**,
- e ser **impossível de auditar** — ninguém consegue explicar por que o sistema fez o que fez.

Isso trava a adoção corporativa: compliance, testes, debugging e orçamento exigem **previsibilidade**.

### A tese do PolyRAG: "determinismo onde importa, IA onde faz diferença"

O núcleo do projeto **não** é "usar mais IA em tudo". É o oposto: **usar matemática determinística sempre que possível, e reservar a IA (LLM) só para onde ela é insubstituível.**

| Etapa | Como o PolyRAG decide | Determinístico? |
|---|---|---|
| Detectar que um arquivo é tabela | **Heurística/regex pura** (densidade de vírgulas/números) | ✅ **Sim, 100%** |
| Roteamento semântico (qual RAG?) | **Cosseno + margem** (aritmética pura: mesmo input → mesmo output, bit a bit) | ✅ **Sim, 100%** |
| Cache semântico (hit/miss?) | **Threshold matemático** (cosseno ≥ 0.92) | ✅ **Sim, 100%** |
| Rankear chunks no grafo | **Personalized PageRank** (algoritmo determinístico de álgebra linear) | ✅ **Sim, 100%** |
| Desempatar rota na zona cinzenta | **LLM-as-judge** (limitado e auditado) | ⚠️ IA, restrita ao mínimo |
| Gerar a resposta final | **LLM** (temperatura fixa e baixa) | ⚠️ IA, única parte "criativa" |

**Consequências práticas (o que você vende):**
- **Mesma pergunta → mesma rota → mesma resposta.** Testável, debugável, reproduzível.
- **A matemática é 100% testável** com testes unitários (sem LLM envolvido): cosseno, margem, thresholds, PageRank.
- **Cada decisão é auditável**: o span do OpenTelemetry registra `score_top1`, `score_top2`, `margin`, `decision_stage` — dá para provar *por que* o sistema fez o que fez.
- **Custo e latência previsíveis**: o LLM só é chamado onde é necessário (zona cinzenta + geração), o resto é aritmética em microssegundos.

### Comparação honesta com o ecossistema

| | LangChain | LlamaIndex | FedRAG (acadêmico) | **PolyRAG** |
|---|---|---|---|---|
| Natureza | Framework genérico de LLMs | Biblioteca RAG-first | Framework de pesquisa (privacidade) | **Reference implementation** |
| Roteamento | Agents/function-calling (não-determinístico) | RouterRetriever por keyword/LLM (caixa-preta) | — | **3 estágios: heurística → cosseno+margem → juiz (auditável)** |
| Decisão | Imprevisível | Caixa-preta | — | **Determinística e rastreável** |
| Hardware | Requer CUDA/OpenAI padrão | Requer stack padrão | Requer stack padrão | **AMD/Vulkan/Windows, sem Docker, sem CUDA** |
| Cache | Não built-in | Não built-in | — | **CAG (FAISS em RAM) de fábrica** |
| Transparência | Baixa | Baixa | — | **Matemática e decisões à mostra + OTel** |

> **Nota de precisão:** o FedRAG acadêmico usa "federado" para privacidade entre *organizações*. O PolyRAG usa "federado" no sentido de *múltiplas bases heterogêneas* (relacional + vetorial + grafo) atrás de uma porta unificada. Na terminologia de mercado isso é "multi-store/hybrid RAG" — vale usar esse termo em entrevistas.

### O pitch da entrevista (versão madura e honesta)

> *"Não reivindico inovação algorítmica — reivindico engenharia com um princípio: **determinismo onde importa, IA onde faz diferença**. Os frameworks de mercado delegam as decisões ao LLM, o que torna o sistema imprevisível e inauditável. O PolyRAG roteia por matemática (cosseno + margem), aplica PageRank determinístico no grafo, usa cache semântico por threshold — e restringe o LLM a desempates e à geração final. Resultado: mesma pergunta, mesma resposta, cada decisão com prova nos spans do OpenTelemetry. E tudo roda em hardware democrático: AMD/Vulkan, sem CUDA, sem Docker. Isso é portabilidade e transparência — não algoritmo novo."*

---

## 1. Documentações oficiais (consultar SEMPRE antes de codar cada parte)

| Ferramenta | Documentação | Quando consultar |
|---|---|---|
| llama.cpp (build/Vulkan) | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md | Dia 1 (subir servidores) |
| llama.cpp (multimodal/VLM) | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md | Dia 1 (OCR GLM-OCR) |
| llama.cpp server API | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | Dias 1–4 (endpoints) |
| FastAPI | https://fastapi.tiangolo.com/ | Dias 1, 5 |
| OpenAI Python SDK (cliente p/ llama-server) | https://github.com/openai/openai-python | Dias 1–4 |
| PyYAML | https://pyyaml.org/wiki/PyYAMLDocumentation | Dia 1 (config) |
| FAISS | https://faiss.ai/index.html · https://github.com/facebookresearch/faiss/wiki | Dia 2 |
| NumPy | https://numpy.org/doc/stable/ | Dia 2 (matemática do router) |
| Qdrant | https://qdrant.tech/documentation/ | Dia 3 |
| SQLite (stdlib Python) | https://docs.python.org/3/library/sqlite3.html | Dia 3 |
| pandas | https://pandas.pydata.org/docs/ | Dia 3 |
| networkx (grafo + PageRank) | https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html | Dia 4 |
| HippoRAG (paper/repo de referência — NÃO instalar) | https://github.com/OSU-NLP-Group/HippoRAG · **paper HippoRAG 2:** https://arxiv.org/pdf/2502.14802 | Dia 4 |
| OpenTelemetry Python | https://opentelemetry.io/docs/languages/python/ · https://opentelemetry-python.readthedocs.io/en/latest/ | Dia 1 |
| semantic-router (referência conceitual — NÃO instalar) | https://semantic-router.readthedocs.io/en/latest/ | Dia 2 (design do router) |
| SvelteKit 2 | https://svelte.dev/docs/kit/introduction | Dia 5 |
| TailwindCSS v4 + SvelteKit | https://tailwindcss.com/docs/installation/framework-guides/sveltekit | Dia 5 |
| marked | https://www.npmjs.com/package/marked | Dia 5 |
| uv | https://docs.astral.sh/uv/ | Setup |

---

## 2. Arquitetura geral (o mapa do prédio)

```
                        ┌──────────────────────────────────────────────┐
                        │            FRONTEND (SvelteKit)              │
                        │   Chat  │  Painel de Observabilidade (OTel)  │
                        └──────────────┬───────────────▲───────────────┘
                            REST/WS    │               │  WS spans JSON
                        ┌──────────────▼───────────────┴───────────────┐
                        │        BACKEND FastAPI  (/api/v1/)           │
                        │                                              │
                        │  ┌──────────┐ ┌───────────┐ ┌─────────────┐  │
                        │  │ CAG      │ │ ROTEADOR  │ │ ORQUESTRADOR│  │
                        │  │ (FAISS   │ │ 3 estágios│ │ (spans OTel │  │
                        │  │  em RAM) │ │ semântico │ │  por etapa) │  │
                        │  └──────────┘ └───────────┘ └─────────────┘  │
                        │         ▼ rota escolhida                     │
                        │  ┌──────────┬────────────┬───────────────┐   │
                        │  │ RAG 1    │ RAG 2      │ RAG 3         │   │
                        │  │ SQLite   │ Qdrant     │ Grafo (PPR)   │   │
                        │  │Text-to-  │ vetorial   │ networkx      │   │
                        │  │  SQL     │            │               │   │
                        │  └──────────┴────────────┴───────────────┘   │
                        └──────────────┬───────────────────────────────┘
                                       │ HTTP (API OpenAI-compatible)
              ┌────────────────────────┼─────────────────────────┐
              │                        │                         │
   ┌──────────▼─────────┐  ┌───────────▼──────────┐  ┌───────────▼─────────┐
   │ llama-server :8080 │  │ llama-server :8081   │  │ llama-server :8082  │
   │ Qwen3-14B          │  │ GLM-OCR (OCR)       │  │ BGE-M3 (embeddings) │
   │ raciocínio/SQL/NER │  │ OCR de imagens       │  │ texto → vetor[1024] │
   └────────────────────┘  └──────────────────────┘  └─────────────────────┘
              ▲                    ▲                          ▲
              └──────────  GPU AMD RX 9060 XT (Vulkan) ───────┘

   INGESTÃO:  data_drop/  →  (imagem? GLM-OCR extrai texto)  →  chunking
             →  roteador decide  →  grava na base certa (SQL / Qdrant / Grafo)
```

---

## 3. BACKEND — o que cada ferramenta faz (explicado fácil)

| Ferramenta | O que é, em linguagem simples | Papel no projeto |
|---|---|---|
| **Python 3.12** | A linguagem do backend. Travamos em 3.12 porque algumas libs ainda não aceitam 3.14. | Base de tudo |
| **uv** | Instalador de pacotes Python ultrarrápido (feito em Rust). Substitui `pip` + `venv` + `pip-tools`. Garante que qualquer pessoa que clone o repo instale *exatamente* as mesmas versões. | Setup reproduzível |
| **FastAPI** | Framework para criar APIs HTTP em Python. Você escreve funções, ele vira endpoints com documentação automática (Swagger). | A "recepção" do sistema |
| **Uvicorn** | O servidor que mantém o FastAPI de pé atendendo requisições, inclusive WebSockets. | O "prédio" onde a recepção trabalha |
| **PyYAML** | Lê arquivos `.yaml` (formato de configuração legível por humanos). | O `config.yaml` controla tudo sem mexer no código |
| **OpenAI SDK** | Cliente Python oficial da OpenAI. O truque: o `llama-server` **finge ser a API da OpenAI**, então usamos esse cliente apontando para `localhost`. Padrão de mercado. | Falar com os 3 modelos de IA |
| **NumPy** | A calculadora científica do Python: vetores, matrizes e operações de álgebra linear rápidas. | Toda a matemática do roteador |
| **FAISS (CPU)** | Biblioteca da Meta para **busca de vetores parecidos**. Você dá um vetor, ela acha os mais próximos entre milhares, em microssegundos. | CAG (cache) + índice do roteador |
| **Qdrant** | Um **banco de dados feito só para vetores**. Guarda os pedaços de texto como pontos num espaço de 1024 dimensões e busca por proximidade. Roda como programa standalone (sem Docker). | RAG 2 (memória de textos) |
| **SQLite** | Banco de dados relacional inteiro num **arquivo único**, já vem dentro do Python. Zero instalação. | RAG 1 (memória de tabelas) |
| **pandas** | Excel dentro do Python: lê CSV/XLSX, transforma em tabelas, joga no SQLite. | Ingestão de planilhas |
| **networkx** | Biblioteca de **grafos** (nós e arestas) em Python. Já traz o PageRank implementado. | RAG 3 (memória de conexões) |
| **OpenTelemetry (OTel)** | O **padrão da indústria para observabilidade**. Funciona como o rastreio dos Correios: cada etapa do pipeline gera um "registro de passagem" (span) com horário de entrada, saída e etiquetas (rota escolhida, nota de confiança, cache hit/miss). | O painel de vidro do sistema |
| **python-multipart / httpx / websockets** | Upload de arquivos, chamadas HTTP de health-check e canais de tempo real. | Utilidades da API |

### As 3 IAs locais (todas rodam no `llama-server` via Vulkan)

| Modelo | Analogia | O que faz |
|---|---|---|
| **Qwen3-14B** (Q4_K_M, ~9.5GB VRAM) | O **cérebro raciocinador** | Traduz pergunta → SQL, desempata rotas duvidosas, extrai entidades/relações dos textos (NER), escreve as respostas finais |
| **GLM-OCR** (Q8_0 + mmproj Q8_0, ~2.3GB VRAM) | Os **olhos** | Modelo especializado em OCR: lê foto de tabela/documento escaneado e devolve o texto (incluindo tabelas em Markdown) |

> **Orçamento de VRAM (medido nos arquivos reais):** Qwen3-14B Q4_K_M (~9.5GB) + GLM-OCR Q8_0 + mmproj (~2.3GB) + BGE-M3 Q8_0 (~0.6GB) ≈ **12.4GB / 16GB** ✅. O GLM-OCR tem só ~830M parâmetros — os 3 modelos ficam **sempre de pé**. O modo "OCR sob demanda" vira uma otimização opcional para máquinas com menos VRAM.
| **BGE-M3** (~0.6GB VRAM) | O **tradutor de significados** | Converte qualquer texto num vetor de 1024 números (o "GPS semântico"). Multilingue — funciona em PT-BR |

> **Vulkan** é a API gráfica universal (AMD, NVIDIA, Intel). É o que permite rodar IA na sua AMD **sem CUDA** (que é coisa exclusiva da NVIDIA). O llama.cpp tem build oficial pronto para Vulkan no Windows.

---

## 4. A MATEMÁTICA DO PROJETO (revisão do zero, com exemplos)

> Esta é a seção para reler antes de entrevistas. Cada conceito tem: *o que é → a fórmula → exemplo numérico → onde usamos no projeto.*

### 4.1 Vetores e Embeddings — "significado vira geometria"

**Conceito:** um vetor é só uma lista de números: `[0.2, -0.5, 0.9, ...]`. Um **embedding** é o vetor que uma IA (BGE-M3) gera para um texto. A mágica: textos com significados parecidos viram vetores que **apontam para direções parecidas** no espaço.

- Nosso espaço tem **1024 dimensões** (cada texto vira uma lista de 1024 números).
- Analogia: é como se cada texto ganhasse uma coordenada GPS num "mapa de significados". "Vendas do nordeste" e "faturamento regional" ficam em bairros vizinhos; "receita de bolo" fica longe.

**Onde usamos:** roteador semântico, cache CAG, RAG 2 (Qdrant), RAG 3 (match de entidades).

### 4.2 Norma L2 e Normalização — "padronizar o tamanho das flechas"

**Conceito:** a **norma L2** é o *comprimento* do vetor (generalização do Teorema de Pitágoras):

```
‖v‖ = √(v₁² + v₂² + ... + vₙ²)
```

**Normalizar** = dividir o vetor pelo seu comprimento → ele passa a ter comprimento **exatamente 1** (vetor unitário). Sobram só as *direções*, eliminando o "tamanho".

**Por que importa:** depois de normalizar, **similaridade de cosseno = produto escalar** (conta muito mais barata). É o truque que usamos no FAISS e no router.

### 4.3 Produto Escalar e Similaridade de Cosseno — "medindo o ângulo entre significados"

**Produto escalar (dot product):** multiplique posição a posição e some:

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

**Similaridade de cosseno:** o cosseno do ângulo entre os dois vetores:

```
                a · b
cos(θ) = ─────────────────
              ‖a‖ × ‖b‖
```

- `cos = 1` → mesma direção (significados idênticos)
- `cos = 0` → perpendiculares (assuntos não relacionados)
- Resultado sempre entre -1 e 1 (na prática, com embeddings de texto, entre 0 e 1).

**Exemplo numérico (2 dimensões):**
```
a = [1, 2]   b = [2, 4]
a · b = 1×2 + 2×4 = 10
‖a‖ = √(1+4) = 2.236      ‖b‖ = √(4+16) = 4.472
cos = 10 / (2.236 × 4.472) = 1.0   → vetores alinhados!
```

**Por que cosseno e não distância comum (euclidiana)?** Porque um texto longo gera vetor "maior", mas o *assunto* está na **direção**, não no tamanho. O cosseno ignora tamanho e mede só a direção.

**Onde usamos:** literalmente todo o projeto — router, cache, Qdrant, match de entidades no grafo.

### 4.4 Margem de confiança — "quão certo está o roteador?"

**Conceito:** o roteador calcula o cosseno da entrada contra as frases de exemplo de cada rota e tira o melhor score por rota. Depois olha as duas rotas mais votadas:

```
margem = score_top1 − score_top2
```

- Margem **grande** → decisão clara ("vendas no trimestre" → relational com folga).
- Margem **pequena** → dúvida legítima → chamamos o **LLM juiz** para desempatar (Passo 3).

**Regra de decisão (calibrável no `config.yaml`):**
```
SE score_top1 ≥ 0.62  E  margem ≥ 0.08  → rota top1 (confiante)
SE score_top1 < 0.35                     → fallback: vectorial
SENÃO                                    → zona cinzenta → LLM juiz
```

Isso é um **classificador por vizinho mais próximo (1-NN)** com regra de rejeição — conceito clássico de ML, ótimo para explicar em entrevista.

### 4.5 Índices de busca vetorial — força bruta vs. grafos (FAISS vs. Qdrant/HNSW)

**Busca exata (FAISS `IndexFlatIP`):** compara a query com **todos** os N vetores, um a um, via produto interno. Parece ingênuo, mas com N pequeno (nosso cache: até 10 mil vetores) leva **microssegundos** porque o FAISS usa instruções SIMD da CPU (faz várias multiplicações por ciclo de clock).

**Busca aproximada (HNSW, dentro do Qdrant):** quando N chega a milhões, força bruta não escala. O HNSW constrói um **grafo em camadas**:
- Camada do topo: poucos nós, ligando "bairros" distantes (atalhos de longa distância).
- Camadas de baixo: cada vez mais nós, conexões cada vez mais curtas.
- A busca começa no topo ("que região do mapa?") e desce refinando ("que bairro? que rua?").

**Analogia:** encontrar uma casa sem GPS: primeiro você vai para a cidade certa (rodovia), depois o bairro (avenida), depois a rua. O HNSW faz isso no espaço de 1024 dimensões. Custo: ~log(N) em vez de N.

**Trade-off clássico:** trocamos **exatidão por velocidade** (recall ~99% com busca 100x mais rápida). Ótimo ponto de conversa técnica.

### 4.6 PageRank e Personalized PageRank — o coração do RAG 3

**Grafo:** conjunto de **nós** (entidades: "Fornecedor A", "Lei 13.709", "Processo X") e **arestas** (relações: "fornece_para", "regulamenta", "depende_de"). Direção importa → grafo dirigido.

**PageRank clássico (o algoritmo original do Google, 1998):** a importância de um nó vem da importância de quem aponta para ele. Modele um **passeio aleatório**: um turista andando pelo grafo; a cada passo ele segue uma aresta aleatória saindo do nó atual. A fração de tempo que ele passa em cada nó, no longo prazo, é o PageRank.

```
PR(nó) = (1−d)/N  +  d × Σ  PR(vizinho) / saídas(vizinho)

d = 0.85 (fator de amortecimento)
(1−d) = chance de "teletransporte" para um nó qualquer (não ficar preso em becos)
```

Matematicamente, isso é encontrar o **autovetor principal da matriz de transição** — ou, em linguagem simples: rodar a simulação até as proporções estabilizarem (**distribuição estacionária de uma cadeia de Markov**). Na prática a networkx resolve iterando ~100 vezes.

**Personalized PageRank (PPR) — nossa versão:** o teletransporte **não é mais para qualquer nó** — ele sempre volta para os **nós-semente** (as entidades detectadas na pergunta do usuário). Resultado: o ranking mede *"relevância em relação a ESTA pergunta"* e não importância global.

**Por que isso dá multi-hop:** a "importância" escorre pelas arestas por vários saltos. Se a pergunta menciona "Fornecedor A", o PageRank espalha crédito para "Contrato Y" que depende dele, e dali para a "Cláusula Z" — 2 saltos de raciocínio, sem nenhuma regra programada. É raciocínio emergente por propagação no grafo.

### 4.7 Quantização — "como 14 bilhões de parâmetros cabem em 16GB"

Os "números" dentro de um modelo de IA normalmente são de 16 bits (FP16). A **quantização** os comprime para ~4 bits (Q4_K_M): 4x menos memória, com perda mínima de qualidade.

```
Qwen3-14B em FP16:  14B × 2 bytes   ≈ 28 GB  → não cabe
Qwen3-14B em Q4:    14B × 0.5 byte  ≈  8–9 GB → cabe com folga ✅
```

É como comprimir uma foto RAW (pesadíssima) para JPEG de alta qualidade: o olho humano (aqui, a qualidade das respostas) quase não percebe.

### 4.8 SQL como álgebra relacional (bônus de entrevista)

SQL é **teoria de conjuntos disfarçada**: `WHERE` é seleção (σ), `SELECT colunas` é projeção (π), `JOIN` é produto cartesiano filtrado (⨝), `GROUP BY` particiona conjuntos e aplica funções de agregação. Citar isso mostra que o Text-to-SQL do RAG 1 tem fundamento matemático, não é só "gerar código".

---

## 5. AS 3 BASES DO FEDERATED RAG (o coração científico)

**A taxonomia que guia a separação (pitch):** cada RAG existe porque cobre um tipo de dado
fundamentalmente diferente, com a estrutura de recuperação certa pra cada um — não é "3 bancos por
3 bancos", é "3 formatos de informação, 3 formas certas de consultar cada um".

| RAG | Categoria | Por quê é diferente dos outros |
|---|---|---|
| RAG 1 (SQLite) | **Estruturado / relacional** | Schema fixo, linhas e colunas — definição formal de "relacional" em banco de dados. Agregação exata (soma, média) é determinística: SQL certo = número certo, sem LLM fazendo conta. |
| RAG 2 (Qdrant) | **Não-estruturado** | Texto livre, sem schema, significado captado por similaridade de embedding — não faz sentido tentar tabular ou modelar como grafo. |
| RAG 3 (grafo) | **Conectado / relacionamentos explícitos** | Nós e arestas, foco em multi-hop e dependências (A afeta B afeta C) — note que "conectado" aqui é deliberadamente **diferente** de "relacional" (RAG 1): o grafo não tem schema fixo de linhas/colunas, tem topologia de relações. |

### RAG 1 — Relacional (Text-to-SQL) · Dia 3
- **Dados:** CSV, XLSX, tabelas extraídas de imagens pelo GLM-OCR.
- **Ferramentas:** pandas (lê planilhas) → SQLite (guarda) → Qwen3-14B (traduz pergunta → SQL) → validação read-only (só `SELECT`) → execução → resposta com tabela Markdown.
- **Matemática:** álgebra relacional (4.8). Detecção de "isso é tabela?" usa a heurística do Dia 1 (densidade de vírgulas/números — estatística simples).
- **Doc:** https://docs.python.org/3/library/sqlite3.html · https://pandas.pydata.org/docs/

### RAG 2 — Vetorial (Multi-Modal) · Dia 3
- **Dados:** textos livres, artigos, documentação, e o texto extraído por OCR (GLM-OCR) das imagens.
- **Ferramentas:** BGE-M3 (embeddings) → Qdrant (armazena e busca por HNSW).
- **Matemática:** embeddings (4.1) + cosseno (4.3) + HNSW (4.5).
- **Doc:** https://qdrant.tech/documentation/

### RAG 3 — GraphRAG (HippoRAG 2 reimplementado) · Dia 4
- **Dados:** regras de negócio, compliance, dependências lógicas — perguntas que exigem "saltos" (A afeta B que afeta C).
- **Pipeline (conforme o paper HippoRAG 2, OSU-NLP):**
  1. **NER/OpenIE:** Qwen3-14B extrai triplas `(sujeito, relação, objeto)` de cada chunk.
  2. **Grafo:** networkx monta nós (entidades + chunks) e arestas (relações). Persistido em JSON.
  3. **Busca:** entidades da pergunta → seeds → `nx.personalized_pagerank` → chunks mais relevantes → resposta.
- **Matemática:** grafos + Personalized PageRank (4.6) + cosseno para o match das entidades (4.3).
- **Doc:** https://networkx.org/documentation/stable/ · referência: https://github.com/OSU-NLP-Group/HippoRAG · **paper (matemática e conceito):** https://arxiv.org/pdf/2502.14802

> ⚠️ **Decisão arquitetural registrada:** a biblioteca oficial `hipporag` foi descartada porque exige `torch==2.5.1` + `vllm` (Linux/CUDA-only) — viola as restrições do projeto. Reimplementamos o algoritmo do paper com networkx + nosso llama-server. Isso é um **plus** de portfólio, não uma gambiarra.

### O ROTEADOR SEMÂNTICO (engine própria) — Dia 2 (ingestão) e Dia 4 (busca)

3 estágios, do mais barato ao mais caro:

| Estágio | Quando | Como | Matemática |
|---|---|---|---|
| **1. Heurística** | só na ingestão | regex/estatística: densidade de vírgulas, % de números, padrão de header → score ≥ 0.8 ⇒ `relational` | estatística descritiva |
| **2. Embeddings** | ingestão e chat | cosseno contra as frases de exemplo de cada rota (definidas no `config.yaml`) + regra de margem | 4.1, 4.3, 4.4 |
| **3. Juiz LLM** | só zona cinzenta | Qwen3-14B escolhe entre as 2 rotas mais votadas, com few-shot | — (custo alto, uso raro) |

Inspirado conceitualmente na lib `semantic-router` (https://semantic-router.readthedocs.io/en/latest/), mas implementado por nós com NumPy/FAISS — a matemática fica visível e testável (`tests/test_router_math.py`).

#### Como o roteador decide, na ingestão (passo a passo)

1. **Âncoras pré-computadas:** na inicialização, o BGE-M3 converte cada frase de exemplo do `config.yaml` em um vetor. Cada rota vira um conjunto de âncoras no espaço de 1024 dimensões.
2. **Chunk chega** (texto, OCR de imagem ou linhas de planilha) → vira um vetor.
3. **Mede similaridade:** cosseno entre o vetor do chunk e **todas** as âncoras. Para cada rota, guarda a âncora mais parecida.
4. **Ordena e calcula a margem:** `top1` = rota mais parecida, `top2` = segunda, `margem = top1 − top2`.
5. **Decide:**
   - `top1 ≥ tau_high (0.62)` **e** `margem ≥ delta (0.08)` → vai para `top1` ✅ (geométrico, custo ~0)
   - `top1 < tau_low (0.35)` → fallback `vectorial`
   - **senão (zona cinzenta)** → **LLM-as-judge**: Qwen3-14B lê o chunk + as descrições das rotas e escolhe (custa 1 chamada de LLM)
6. **Grava** o chunk na base da rota escolhida. Tudo registrado no span: `router.route`, `router.score_top1`, `router.score_top2`, `router.margin`, `router.decision_stage`.

> **O mesmo componente serve os dois momentos:** na **ingestão** o input é o *conteúdo do chunk* ("onde guardo?"); na **busca** o input é a *pergunta do usuário* ("onde procuro?"). Mesmas âncoras, mesma matemática, uma engine só. É por isso que as frases-âncora do `config.yaml` são tão importantes: elas *treinam* o roteador sem escrever código.

#### Perguntas compostas: roteamento multi-rota (fan-out + RRF)

Algumas perguntas exigem dados de mais de uma base:

> *"Qual foi o total de vendas em março, e isso fere alguma política de compliance?"*
> → `total de vendas` está no **RAG 1 (SQL)** · `política de compliance` está no **RAG 3 (grafo)**

**Procedimento (determinístico na decisão, paralelo na execução):**

1. **Detectar que é composta** (regra matemática, mesmo princípio da margem):
   - Se `top2 ≥ tau_multi` (novo threshold no config, ex: **0.5**) → ativa **fan-out** (consulta as top-N rotas).
   - Na zona cinzenta, o LLM-as-judge responde **"quais rotas"** (lista JSON), não "qual rota".
2. **Fan-out em paralelo:** `asyncio.gather` consulta as N rotas selecionadas; cada RAG retorna seus top-k com scores.
3. **Fusão por Reciprocal Rank Fusion (RRF)** — matemática clássica de IR, **determinística** e que ignora a escala de cada base:

```
score_final(chunk) = Σ [ 1 / (k + rank_na_rota) ]        k = 60 (constante clássica)
                     rotas

Ex: chunk rank #2 no SQL + rank #5 no grafo
    → 1/(60+2) + 1/(60+5) = 0.0161 + 0.0154 = 0.0315
```

4. **LLM final sintetiza** o contexto multi-fonte com tags de origem (`[SQL]`, `[GRAFO]`) e cita as fontes.
5. **Custo controlado:** fan-out só quando o threshold indicar; span registra `router.routes=[relational, graph]`, `router.fan_out=true` e um span por rota consultada. O CAG continua por pergunta inteira.

> **Por que RRF e não somar os scores crus?** cada RAG pontua em escala diferente (SQL = valor agregado, Qdrant = cosseno, grafo = PageRank). Somar seria comparar bananas com laranjas; RRF usa só os **ranks**, tornando a fusão agnóstica de escala — e é pura aritmética (determinística, testável). Excelente ponto de entrevista.

### O CAG — Cache Semântico (Dia 4, no caminho de busca)

Pergunta repetida (ou *quase* repetida: "total de vendas" ≈ "soma das vendas") não deve pagar o custo do pipeline inteiro. Guardamos `(embedding da pergunta, resposta)` num índice FAISS em RAM. Se `cosseno ≥ 0.92` com algo já respondido → devolve na hora (**latência ~0ms, span com `cache.hit=true`**). Mesma matemática de 4.3, outra aplicação.

### AVALIAÇÃO OFFLINE — recall, precisão e exatidão de resposta (Dia 11)

> **Recap crucial:** recall real exige *ground truth* — por isso é uma métrica de **benchmark offline** (dataset com resposta certa), não de produção em tempo real.

**`scripts/evaluate.py`** roda um benchmark de **50–100 perguntas**, cada uma com seu **golden set**: `{pergunta, rota_certa, chunks_relevantes[], resposta_esperada}`. Para cada pergunta:

```
retrieval recall@k    = |chunks relevantes ∩ top-k recuperados| / |chunks relevantes|
retrieval precision@k = |chunks relevantes ∩ top-k recuperados| / k
router_accuracy       = (perguntas com rota == rota_certa) / total
answer_correctness    = resposta gerada vs resposta esperada (ver rubric abaixo)
```

**O golden set é dividido por domínio para você ver recall por RAG:**
- **RAG 1 (SQL):** resposta esperada é um **valor exato** ("total de março = 4.523"). Como a resposta vem da execução do SQL, `answer_correctness` aqui é **determinística** — se o Text-to-SQL acertou a query, acertou o número. Atinge valores altos (90%+).
- **RAG 2 (vetorial):** resposta "certa" é texto livre → `answer_correctness` via **LLM-as-judge com rubrica** (0/1: a resposta contém a informação do gold?) + cosseno contra a resposta esperada como apoio.
- **RAG 3 (grafo/multi-hop):** além do recall@k, checar se a **cadeia de entidades** (A→B→C) apareceu na resposta.

**Exportação para OTel:** o benchmark roda num serviço próprio (`service.name = polyrag-eval`) e cada rodada emite spans:
`span "eval.recall"` com attributes `eval.recall@1`, `eval.recall@5`, `eval.recall@10`, `eval.precision@10`, `eval.router_accuracy`, `eval.answer_correctness`, `eval.route` — aparecem no painel como qualquer trace.

**Metas realistas (para calibrar expectativa):**
| Métrica | Meta | Por quê |
|---|---|---|
| recall@10 (sintético, controlado) | 90–98% | top-k generoso + chunks discriminativos |
| recall@1 | 60–85% | achar o chunk certo de primeira é mais duro |
| router_accuracy | 90–97% | cosseno+margem bem calibrado |
| answer_correctness (RAG 1/SQL) | 90%+ | **determinística** — acertou SQL, acertou número |
| answer_correctness (RAG 2/3, texto) | 70–90% | depende da geração do LLM local; 99% irrealista |

> **Nota de maturidade:** 99% de recall e resposta certa são objetivos diferentes. `recall` = "a informação estava no contexto?"; `answer_correctness` = "a resposta final está certa?". Sempre reporte os dois, e o top-k junto (recall@1 ≠ recall@10). Dataset construído por você infla os números — a honestidade é reportar a curva e o contexto do golden set.

#### O que cada métrica responde de verdade (recall vs resposta certa)

> **Insight central:** recall mede a **recuperação** ("a matéria-prima certa entrou no contexto?"). `answer_correctness` mede o **resultado final** ("a resposta certa chegou ao usuário?"). Você precisa dos DOIS porque juntos formam o **diagnóstico**:

```
        ┌────────────────────────────────────────────────┐
 resposta errada?                                       │
        ▼                                                │
 RECALL baixo? ──► "a informação NEM entrou no contexto" → problema no RETRIEVAL
 RECALL alto?  ──► "a info estava lá, mas o LLM errou"   → problema na GERAÇÃO
```

- `recall` baixo + resposta errada → o defeito é na **recuperação** (não achou).
- `recall` alto + resposta errada → o defeito é na **geração** (leu e errou).
- Ambos altos → pipeline saudável; a confiança do sistema está comprovada.

**Como medir `answer_correctness` aceitando "mesma informação, outra redação":** o LLM-as-judge com **rubrica binária** (temperatura 0 para máxima determinismo):

```
Prompt do juiz:
"Você é um avaliador de respostas. Compare a RESPOSTA GERADA com a RESPOSTA ESPERADA.
Responda apenas JSON: {"correct": 0|1, "justification": "..."}
- correct = 1 se a resposta gerada transmite a MESMA informação da esperada,
  mesmo que escrita com palavras diferentes.
- correct = 0 se faltou informação essencial ou se houver informação contraditória."
```

É essa métrica que responde à pergunta *"a resposta certa chegou ao usuário, mesmo reescrita?"* — e é ela que vale destacar nos resultados de pesquisa, junto com o recall@k que explica o *porquê*.

#### Recursos de referência para apresentar (dia da apresentação)

| Paper/ferramenta | Link | Para citar |
|---|---|---|
| HippoRAG 2 — *From RAG to Memory: Non-Parametric Continual Learning for LLMs* | https://arxiv.org/pdf/2502.14802 | Base do RAG 3: NER + grafo + **Personalized PageRank** para multi-hop |
| Repo oficial OSU-NLP | https://github.com/OSU-NLP-Group/HippoRAG | Implementação de referência (que adaptamos sem torch/vLLM) |

---

## 6. O MOTOR E A API (o produto open-source)

- **Motor:** `llama.cpp` build oficial Vulkan (`b10453`) — 3 processos `llama-server` (portas 8080/8081/8082), cada um servindo um modelo, todos falando **API OpenAI-compatible** (`/v1/chat/completions`, `/v1/embeddings`). O GLM-OCR é leve (~2.3GB), então os 3 cabem juntos nos 16GB; um modo "OCR sob demanda" fica documentado como otimização opcional.
- **API:** FastAPI, rotas sob `/api/v1/`, Swagger automático em `/docs`, WebSockets para streaming de tokens e de telemetria.
- **Observabilidade:** OTel SDK no FastAPI. Cada etapa (`pipeline.cache`, `pipeline.router`, `pipeline.rag.*`, `pipeline.llm`) gera **spans** com atributos (`router.route`, `router.margin`, `cache.hit`...). Um **exporter customizado** publica os spans em JSON via WebSocket para o frontend — sem Grafana, sem Collector, sem servidor extra.

### Contrato da API v1

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/v1/health` | status de llm/ocr/embeddings/qdrant |
| POST | `/api/v1/chat` | `{message}` → `{answer, route, trace_id, sources[]}` |
| WS | `/api/v1/chat/stream` | resposta token a token + eventos do pipeline |
| POST | `/api/v1/ingest` | upload direto de arquivo |
| GET | `/api/v1/ingest/jobs` | status das ingestões |
| WS | `/api/v1/telemetry/stream` | spans OTel em tempo real (alimenta o painel) |
| GET | `/api/v1/telemetry/traces` | últimos traces (ring buffer) |
| DELETE | `/api/v1/cache` | limpa o CAG |

---

## 7. FRONTEND (SvelteKit) — a "Glass-Box UI" (Dia 5)

| Ferramenta | O que é | Papel |
|---|---|---|
| **SvelteKit 2** | Framework web que compila para JS enxuto (sem runtime pesado) | Esqueleto da UI |
| **TailwindCSS v4** | CSS por classes utilitárias, direto no HTML | Visual rápido e consistente |
| **marked** | Converte Markdown → HTML no navegador | Renderiza respostas, tabelas SQL e citações |

**Layout em 2 colunas:**
- **Esquerda — Chat:** streaming token a token (WS), Markdown renderizado, tabelas SQL formatadas.
- **Direita — Painel de Observabilidade:** consome o WS de telemetria e desenha, ao vivo, a **linha do tempo dos spans** (Cache → Router → RAG → LLM) com latência de cada etapa, a rota escolhida com sua margem, e hit/miss do cache. É a "caixa de vidro": o usuário *vê a matemática acontecendo*.

---

## 8. Estrutura de pastas e setup

```
polyrag/
├── .github/workflows/ci.yml    # Dia 6 (CI)
├── .gitignore · .python-version (3.12) · AGENTS.md · README.md
├── IMPLEMENTATION_PLAN.md      # este arquivo
├── config.yaml                 # TODOS os parâmetros (modelos, thresholds, rotas)
├── pyproject.toml · uv.lock
├── data_drop/                  # 📥 hot folder de ingestão
├── data/
│   ├── sql/relacional.db       # RAG 1
│   ├── qdrant/                 # storage do Qdrant
│   ├── graph/graph.json        # RAG 3
│   └── processed/              # arquivos já ingeridos
├── models/                     # GGUFs (~14GB, gitignored)
├── runtime/
│   ├── llama/                  # llama-server.exe (Vulkan)
│   └── qdrant/qdrant.exe
├── scripts/
│   ├── start_llm_servers.ps1   # sobe os 3 servidores (LLM + OCR + embeddings)
│   ├── start_qdrant.ps1 · start_backend.ps1 · stop_all.ps1
├── backend/app/
│   ├── main.py                 # app factory + lifespan + instrumentação OTel
│   ├── core/                   # config.py · telemetry.py (exporter WS) · llama_client.py
│   ├── api/v1/                 # health · chat · ingest · telemetry · router agregador
│   ├── pipeline/               # orchestrator · heuristics · router · cache
│   ├── rags/                   # base.py · relational.py · vectorial.py · graph.py
│   ├── ingest/                 # watcher · loaders · ocr · chunking
│   └── schemas/                # DTOs pydantic (chat, ingest, telemetry)
├── frontend/                   # SvelteKit (Dia 5)
└── tests/                      # test_router_math · test_cache · test_api_integration
```

**Setup (PowerShell, na raiz):**
```powershell
# 1. pastas
$dirs = @('.github/workflows','data_drop','data/sql','data/qdrant','data/graph','data/processed','models','runtime/llama','runtime/qdrant','scripts','backend/app/core','backend/app/api/v1','backend/app/pipeline','backend/app/rags','backend/app/ingest','backend/app/schemas','frontend','tests')
$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

# 2. python + deps
Set-Content .python-version "3.12"
uv venv --python 3.12
uv add fastapi==0.141.1 "uvicorn[standard]==0.52.3" pyyaml==6.0.3 openai numpy faiss-cpu==1.15.0 qdrant-client==1.19.0 networkx pandas openpyxl opentelemetry-sdk==1.44.0 opentelemetry-instrumentation-fastapi python-multipart httpx
uv add --dev pytest pytest-asyncio ruff

# 3. llama.cpp Vulkan (b10453)
Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/b10453/llama-b10453-bin-win-vulkan-x64.zip" -OutFile runtime/llama.zip
Expand-Archive runtime/llama.zip runtime/llama; Remove-Item runtime/llama.zip
.\runtime\llama\llama-server.exe --list-devices   # deve listar a RX 9060 XT

# 4. Qdrant standalone (v1.19.0)
Invoke-WebRequest "https://github.com/qdrant/qdrant/releases/download/v1.19.0/qdrant-x86_64-pc-windows-msvc.zip" -OutFile runtime/qdrant.zip
Expand-Archive runtime/qdrant.zip runtime/qdrant; Remove-Item runtime/qdrant.zip

# 5. modelos GGUF (~14GB — pode deixar baixando)
uv tool install huggingface-hub   # fornece o comando 'hf'
hf download Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf --local-dir models
hf download ggml-org/GLM-OCR-GGUF GLM-OCR-Q8_0.gguf mmproj-GLM-OCR-Q8_0.gguf --local-dir models
hf download gpustack/bge-m3-GGUF bge-m3-Q8_0.gguf --local-dir models
```

**`config.yaml` (governa o framework — inclusive as âncoras do roteador):**

```yaml
server:
  host: 127.0.0.1
  port: 8000

llama:
  bin_path: runtime/llama/llama-server.exe
  llm:            # sempre de pé (responde perguntas)
    model_path: models/Qwen3-14B-Q4_K_M.gguf
    host: 127.0.0.1
    port: 8080
    ctx_size: 16384
    n_gpu_layers: -1
    temperature: 0.2
  ocr:            # GLM-OCR ~830M params (~2.3GB VRAM) — fica de pé
    model_path: models/GLM-OCR-Q8_0.gguf
    mmproj_path: models/mmproj-GLM-OCR-Q8_0.gguf
    host: 127.0.0.1
    port: 8081
    ctx_size: 8192
    n_gpu_layers: -1
  embeddings:
    model_path: models/bge-m3-Q8_0.gguf
    host: 127.0.0.1
    port: 8082
    ctx_size: 8192
    n_gpu_layers: -1
    dimensions: 1024

qdrant:
  bin_path: runtime/qdrant/qdrant.exe
  host: 127.0.0.1
  port: 6333
  collection: polyrag_docs
  distance: Cosine
  vector_size: 1024

paths:
  data_drop: data_drop
  processed: data/processed
  sqlite_db: data/sql/relacional.db
  graph_store: data/graph/graph.json

# ---------- Âncoras do Roteador (treinam o roteador sem escrever código) ----------
router:
  tau_high: 0.62
  tau_low: 0.35
  delta_margin: 0.08
  llm_judge_enabled: true
  routes:
    relational:
      description: "Dados tabulares, planilhas, finanças, logs estruturados"
      utterances:
        - "quantas vendas tivemos em março por região"
        - "qual o total de receita do trimestre"
        - "liste os 10 produtos mais vendidos"
        - "média salarial por departamento"
    vectorial:
      description: "Texto não estruturado, narrativas, documentos, artigos"
      utterances:
        - "resuma o documento sobre história da empresa"
        - "o que o artigo diz sobre mudanças climáticas"
        - "quais os pontos principais do relatório narrativo"
    graph:
      description: "Regras de negócio, compliance, dependências, multi-hop"
      utterances:
        - "se o fornecedor A falhar, quais contratos são afetados"
        - "quais políticas se aplicam a transferências internacionais"
        - "rastreie as dependências desse processo de aprovação"

cache:
  enabled: true
  similarity_threshold: 0.92
  max_entries: 10000

ingest:
  watch_interval_s: 2.0
  chunk_size: 512
  chunk_overlap: 64
  ocr_prompt: "Extraia todo o texto visível desta imagem. Reproduza tabelas em Markdown."
  vision_extensions: [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
  text_extensions: [".txt", ".md"]
  table_extensions: [".csv", ".xlsx"]

telemetry:
  enabled: true
  service_name: polyrag
  ring_buffer_size: 500
```

---

## 9. ROADMAP — 2 semanas (núcleo na semana 1, consolidação na semana 2)

### SEMANA 1 — Construção (seguindo a ordem REAL do pipeline de ingestão)

> A ordem abaixo é a do fluxo de dados da aplicação (do arquivo que chega até a resposta), exatamente como você pediu. Antes do Dia 1, conclua o **Setup** da seção 8 (llama.cpp Vulkan, Qdrant, venv, modelos). Cada dia é testável de forma isolada via testes/CLI, mesmo sem o restante pronto.

| Dia | Missão | Entregável | Matemática/conceito do dia |
|---|---|---|---|
| **1** | **Ingestão de conteúdo** — o arquivo chega na `data_drop` | watcher detecta o arquivo; separa imagens de textos/planilhas; **GLM-OCR** extrai o texto das imagens; chunking | — (orquestração + OCR) |
| **2** | **Roteador de ingestão** — decidir onde guardar | heurística (é tabular?) + engine de cosseno/margem + LLM-as-judge na zona cinzenta; chunk cai na rota certa com span de decisão | **cosseno, norma L2, margem** (4.2–4.4) |
| **3** | **As 3 bases (RAGs)** — onde guardar e buscar | RAG1 Text-to-SQL (SQLite, validação read-only); RAG2 Qdrant (HNSW); RAG3 grafo (NER + `personalized_pagerank`) | álgebra relacional; **HNSW** (4.5); **PageRank** (4.6) |
| **4** | **Caminho de busca** — responder | roteador de busca (mesma engine do Dia 2); CAG (cache FAISS em RAM, hit/miss); Qwen3-14B como LLM de resposta; orquestrador completo (Cache→Router→RAG→LLM) | reuso de 4.3/4.4 no caminho de query |
| **5** | **API + Frontend** | rotas finais `/api/v1/`, WebSockets; chat com streaming + painel de observabilidade ao vivo (spans OTel) | — (integração) |
| **6** | **Documentação + CI** | README completo, GitHub Actions, testes de integração verdes | — |
| **7** | **Folga planejada** | absorver atrasos; reler a seção 4 e explicar o projeto em voz alta | revisão |

### SEMANA 2 — Consolidação (transformar "funciona" em "impressiona")

| Dia | Missão | Entregável |
|---|---|---|
| **8** | Hardening | testes unitários da matemática (cosseno, margem, PPR) com casos de borda; tratamento de erros (modelo fora do ar, arquivo corrompido) |
| **9** | Dataset de demonstração | 3 conjuntos de dados de exemplo (planilha de vendas, artigos, regras de compliance fictícias) + roteiro de demo com perguntas que ativam cada rota |
| **10** | Polimento da UI | estados vazios, loading, tema escuro, timeline de spans mais legível |
| **11** | Métricas de portfólio | `scripts/evaluate.py` (50–100 perguntas com golden set) → tabela no README: recall@1/5/10, precisão, router_accuracy e answer_correctness por rota; latência p50/p95 por etapa; taxa de cache hit |
| **12** | Documentação final | README com diagrama da arquitetura, seção "A matemática por trás" (resumo da seção 4), GIF/vídeo da demo |
| **13** | Divulgação técnica | post (dev.to/LinkedIn) explicando o projeto em linguagem simples — ensaiar o pitch da seção 0 |
| **14** | Release | tag `v0.1.0` no GitHub, issues abertas com roadmap futuro, buffer final |

### Definition of Done (o projeto está pronto quando):
- [ ] Drop de imagem + CSV + TXT + "regras" na `data_drop` → cada um cai na base certa, com spans provando
- [ ] 3 perguntas (uma por domínio) respondidas corretamente, com rota e margem visíveis no painel
- [ ] Pergunta repetida → cache hit instantâneo
- [ ] Pergunta multi-hop respondida pelo grafo (PPR) com cadeia de entidades exibida
- [ ] `git clone` → setup → demo rodando, só seguindo o README

---

## 10. Glossário rápido (cola de entrevista)

| Termo | Explicação de 1 linha |
|---|---|
| **RAG** | Responder usando documentos recuperados como contexto, em vez de só a "memória" do modelo |
| **Federated/Unified RAG** | Vários tipos de base (SQL, vetores, grafo) atrás de uma única porta de entrada inteligente |
| **Embedding** | Texto convertido em vetor de números que captura o significado |
| **Cosseno** | Mede o ângulo entre dois vetores = quão parecidos são dois significados |
| **Margem** | Diferença entre a 1ª e a 2ª rota mais prováveis = confiança da decisão |
| **CAG** | Cache semântico: pergunta parecida → resposta instantânea, sem recomputar |
| **HNSW** | Índice de grafo em camadas para achar vetores vizinhos em tempo logarítmico |
| **NER / OpenIE** | Extrair entidades (nomes) e relações (triplas sujeito–relação–objeto) de textos |
| **PageRank** | Importância de um nó pela importância de quem aponta para ele (Google, 1998) |
| **Personalized PageRank** | PageRank com teletransporte sempre de volta aos nós da pergunta → relevância contextual multi-hop |
| **Quantização (Q4)** | Comprimir os números do modelo de 16 para ~4 bits: 4x menos VRAM |
| **OTel / Span / Trace** | Rastro cronometrado de cada etapa do pipeline (o "rastreio dos Correios" do sistema) |
| **GGUF** | Formato de arquivo de modelo de IA pronto para o llama.cpp |
| **Vulkan** | API gráfica universal que deixa a AMD rodar IA sem CUDA |

---

## 11. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| OCR (GLM-OCR Q8_0) falhar em tabelas muito complexas | subir p/ `GLM-OCR-f16.gguf` (1.66GB, sem quantização) — ainda leve, só config |
| Apertar VRAM em pico de contexto (ctx 16384 do LLM) | reduzir `ctx_size` do LLM p/ 12288 no config; ou ativar o modo OCR sob demanda |
| Juiz LLM lento | `llm_judge_enabled: false` no config; threshold conservador |
| Escorregar o Dia 4 (o mais denso) | Dia 7 é folga planejada exatamente para isso |
| Versões novas quebrarem algo | tudo pinado no `uv.lock`; docs linkadas na seção 1 |
| `loaders.py` extrai só texto corrido de PDF/DOCX/PPTX, ignorando imagens e tabelas embutidas dentro do arquivo | Desmontar o documento em sub-elementos (texto/imagem/tabela) e rotear cada um pro tratamento certo (imagem → `ocr.py`, tabela → DataFrame → RAG 1) — ver Dia 8+ (hardening) |
