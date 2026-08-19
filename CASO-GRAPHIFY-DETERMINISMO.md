# Caso de Estudo: Graphify — duas alegações mal fundamentadas ("determinismo" e "79x tokens")

> Documento de apoio para a apresentação do PolyRAG. Objetivo: usar o Graphify como
> **contraexemplo** — um projeto que vende uma vantagem técnica (determinismo) sem separar
> corretamente onde essa propriedade realmente vem, ao contrário do PolyRAG, que trata isso como
> tese central e a aplica etapa por etapa (ver seção 0.1 do `CLAUDE.md`).

**Repo:** https://github.com/Graphify-Labs/graphify (skill do Claude Code, PyPI: `graphifyy`)

---

## 1. O que o Graphify é, de fato

Uma skill que lê uma pasta local (código, PDFs, markdown, imagens) e monta um **grafo de
conhecimento persistente**, em vez de indexar tudo em vetores. Pipeline:

```
detect() → extract() → build_graph() → cluster() → analyze() → report() → export()
```

- **Código:** parsing determinístico via **tree-sitter (AST)** — sem LLM, "nothing leaves your
  machine".
- **Docs, PDFs, imagens:** subagentes do Claude extraem conceitos e relações, marcando cada
  aresta como `EXTRACTED` (explícita na fonte), `INFERRED` (dedução razoável) ou `AMBIGUOUS`
  (incerta, fica para revisão humana).
- **Clustering:** Leiden (via `graspologic`) agrupa nós em comunidades; gera "god nodes" e
  "surprising connections".
- **Consulta (`/graphify query`):** faz correspondência **literal de substring** dos termos da
  pergunta contra os *labels* dos nós para achar 1–3 nós-semente, depois faz travessia **BFS**
  (contexto amplo) ou **DFS** (rastrear um caminho específico) com profundidade limitada. O
  subgrafo resultante vira contexto para o LLM responder.

Dependências (`pyproject.toml`): `networkx`, `graspologic`, `tree-sitter*`. **Nenhuma** lib de
vetor/embedding (sem `faiss`, `chromadb`, `sentence-transformers`, etc.) — confirma que a ausência
de vetor é real, não só discurso.

## 2. A alegação central (a frase que vamos questionar)

Do README (branch `v8`), como um dos três bullets de venda do projeto:

> **"Not a vector index. No embeddings, no vector store: a real graph you traverse."**

Um blog de terceiro que cobriu o lançamento resume isso como **"Zero Vector Database Overhead"** —
ou seja, o argumento de fábrica é: sem vetor, sem infraestrutura extra, sem não-determinismo.

## 3. Por que essa alegação é uma falha de raciocínio

O erro é tratar "usa vetor" e "é não-determinístico" como a mesma coisa. Não são.

| Etapa | É determinística? | Por quê |
|---|---|---|
| Gerar embedding de um texto (encoder fixo) | ✅ Sim | Função pura, sem amostragem — mesmo texto → mesmo vetor |
| Similaridade de cosseno / ranking por distância | ✅ Sim | Aritmética pura (produto escalar, norma) |
| Travessia BFS/DFS do grafo do Graphify | ✅ Sim | Também é aritmética/estrutura pura, igual ao cosseno |
| Parsing de código via tree-sitter (AST) | ✅ Sim | Determinístico nos dois projetos — não depende de LLM |
| Extração de conceitos/relações em docs, PDFs, imagens (Graphify) | ❌ Não | Feita por subagentes do Claude — amostragem de LLM |
| Geração da resposta final a partir do subgrafo (Graphify) | ❌ Não | LLM lendo o contexto e escrevendo a resposta |
| Geração da resposta final a partir do contexto recuperado (RAG vetorial) | ❌ Não | Mesmíssimo passo, mesmíssima fonte de variância |

Conclusão: **a recuperação em si (seja por cosseno, seja por travessia de grafo) já era
determinística nos dois mundos.** O vetor nunca foi a origem da imprevisibilidade que o Graphify
alega evitar. A origem real é **qualquer etapa que passe por um LLM com amostragem** — e o
Graphify tem duas dessas etapas (extração semântica + geração da resposta), exatamente como um RAG
vetorial comum. Ou seja: **o grafo do Graphify não é mais determinístico que um índice vetorial
para o mesmo tipo de conteúdo** — só o parsing de código puro (AST) ganha determinismo total,
porque dispensa LLM. Isso é uma vitória de "código vs. LLM", não de "grafo vs. vetor".

Efeito colateral não anunciado: a busca por nó-semente no Graphify é **match literal de substring
no label**. Isso é estritamente mais fraco que embeddings para achar conceitos com vocabulário
diferente ("carro" vs. "automóvel") — exatamente o problema que busca vetorial resolve bem. Trocar
vetor por grafo puro pode ganhar em auditabilidade e raciocínio multi-hop explícito, mas perde
recall semântico difuso — um trade-off real que a documentação do projeto não expõe.

## 4. Contraste com a tese do PolyRAG

O PolyRAG não faz a alegação genérica "somos determinísticos porque não usamos X". Ele separa o
pipeline em etapas e diz exatamente onde a matemática é determinística e onde a IA entra — e
**minimiza deliberadamente** a segunda categoria:

| Etapa | Graphify | PolyRAG |
|---|---|---|
| Roteamento / decisão de onde buscar | Match literal de termo (fraco, mas determinístico) | Cosseno + margem (4.3–4.4 do `ESTUDO-MATEMATICA.md`), auditado via span OTel |
| Zona cinzenta / caso ambíguo | Não existe estágio explícito — decide na marra com o traversal | LLM-as-judge, restrito e documentado como a única exceção |
| Extração de conteúdo não estruturado | LLM (não-determinístico), sem admitir isso na alegação de marketing | GLM-OCR para imagens, span próprio, custo e papel explicitados |
| Geração da resposta final | LLM (implícito, não contabilizado no "somos determinísticos") | LLM, explicitamente reconhecido como "a única parte criativa" |
| Honestidade da alegação | Afirma determinismo geral a partir de "não ter vetor" | Afirma determinismo *por etapa*, com tabela e spans que provam cada uma |

**Ponto para a apresentação:** o Graphify é um bom exemplo de projeto que identifica corretamente
um problema real do mercado (decisões de RAG pouco auditáveis) mas erra a causa raiz — atribui a
imprevisibilidade ao "vetor" quando na verdade ela vem do uso de LLM em etapas de extração e
geração, presentes nos dois modelos. O PolyRAG evita esse erro porque a tese de determinismo é
aplicada e testada **componente a componente** (com testes unitários da matemática, sem LLM
envolvido), não vendida como propriedade emergente de trocar um tipo de índice por outro.

## 5. O segundo exagero: "79x token reduction"

O README e um post de terceiro alardeiam "71.5x" / "79x fewer tokens per query". O código-fonte do
próprio benchmark (`graphify/benchmark.py`, branch `v8`) mostra exatamente o que está sendo medido
— e revela um baseline artificialmente fraco.

```python
corpus_tokens = corpus_words * 100 // 75      # todo o corpus, texto bruto, virado token
...
qt = _query_subgraph_tokens(G, q)             # tamanho do subgrafo (BFS, profundidade 3) em tokens
per_question.append({..., "reduction": round(corpus_tokens / qt, 1)})
```

`_query_subgraph_tokens` acha os nós-semente por **match literal de substring** no label
(`score = sum(1 for t in terms if t in label)`), faz BFS a partir deles e conta os tokens do texto
resultante (labels + arestas).

### Por que esse número não prova o que parece provar

- **O baseline é "colar o corpus inteiro no prompt em toda pergunta".** Não é comparado contra um
  RAG vetorial com top-k, nem contra chunking, nem contra o roteador do PolyRAG — é comparado
  contra *não fazer nenhuma recuperação seletiva*. Qualquer RAG minimamente decente (inclusive o
  RAG 2/Qdrant do PolyRAG, recuperando top-k chunks) chega a uma razão de redução na mesma ordem de
  grandeza contra esse mesmo baseline, porque a vitória vem de "recuperar seletivamente", não de
  "usar grafo em vez de vetor".
- **O custo de construir o grafo fica fora da conta.** A extração semântica (docs/PDFs/imagens)
  roda subagentes de LLM por arquivo — registrado à parte em `graphify-out/cost.json`
  (`input_tokens`/`output_tokens`), nunca somado ao `reduction_ratio`. Em corpora que mudam com
  frequência (`--update` re-extrai), esse custo amortiza mal para poucas consultas.
- **Tokens baratos ≠ resposta certa.** O mesmo ponto do tópico "recall vs. answer_correctness" da
  seção de avaliação do `CLAUDE.md`: um subgrafo de 2000 tokens pode ser barato e ainda assim não
  conter a informação relevante, porque a seleção do nó-semente depende de a pergunta usar as
  mesmas palavras que aparecem nos labels do grafo — sem embeddings, sinônimos e paráfrases não são
  encontrados.

**Conclusão para o pitch:** o "79x" é real como aritmética (recuperar um subgrafo pequeno é mais
barato que colar tudo), mas não é uma comparação válida contra "RAG" em geral — é uma comparação
contra a ausência de qualquer RAG. Instalar o Graphify não traz economia de tokens acima do que
uma arquitetura de recuperação seletiva (vetorial, relacional ou de grafo, como as do PolyRAG) já
entrega. É o mesmo padrão do erro de determinismo da seção 3: uma propriedade real (recuperação
seletiva economiza tokens) é atribuída à escolha errada de causa (grafo em vez de vetor).

## 6. Fontes

- [Graphify README (branch v8)](https://raw.githubusercontent.com/Graphify-Labs/graphify/v8/README.md)
- [Graphify ARCHITECTURE.md](https://raw.githubusercontent.com/Graphify-Labs/graphify/main/ARCHITECTURE.md)
- [Graphify skill.md](https://raw.githubusercontent.com/Graphify-Labs/graphify/main/skill.md)
- [Graphify benchmark.py (branch v8) — metodologia real do "token reduction"](https://raw.githubusercontent.com/Graphify-Labs/graphify/v8/graphify/benchmark.py)
- [pyproject.toml (dependências, confirma ausência de libs de vetor)](https://raw.githubusercontent.com/Graphify-Labs/graphify/main/pyproject.toml)
- [Graph Your Codebase with Graphify: 79× Token Reduction and Zero Vector Database Overhead](https://stevescargall.com/blog/2026/05/graphify--memmachine-79-token-reduction-zero-vector-database/)
- [graphifyy no PyPI](https://pypi.org/project/graphifyy/)
