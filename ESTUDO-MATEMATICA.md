# Estudo de Matemática Aplicada — PolyRAG

> Documento de estudo autocontido: para cada tópico usado no projeto você tem **teoria resumida**,
> uma **analogia** para fixar a intuição, a **fórmula**, um **exemplo prático executável** (Python
> puro / NumPy, sem dependências pesadas) e **onde exatamente** ele aparece no código do PolyRAG.
> Os valores numéricos (thresholds, dimensões, constantes) são os mesmos do `config.yaml` do
> projeto — não são exemplos genéricos.
>
> Como estudar: leia a teoria, tente prever o resultado do exemplo antes de rodar, depois rode de
> verdade (`python3 -c "..."` ou cole num notebook) e compare. No fim de cada tópico há uma
> pergunta de auto-teste; no fim do documento, um checklist geral.

---

## Sumário

1. [Vetores e Embeddings](#1-vetores-e-embeddings--significado-vira-geometria)
2. [Norma L2 e Normalização](#2-norma-l2-e-normalização--padronizar-o-tamanho-das-flechas)
3. [Produto Escalar e Similaridade de Cosseno](#3-produto-escalar-e-similaridade-de-cosseno)
4. [Margem de Confiança — o Roteador como Classificador 1-NN](#4-margem-de-confiança--o-roteador-como-classificador-1-nn)
5. [Busca Vetorial: Força Bruta (FAISS) vs. Aproximada (HNSW/Qdrant)](#5-busca-vetorial-força-bruta-faiss-vs-aproximada-hnswqdrant)
6. [Grafos, PageRank e Personalized PageRank](#6-grafos-pagerank-e-personalized-pagerank)
7. [Quantização de Modelos](#7-quantização-de-modelos)
8. [Álgebra Relacional (SQL como Teoria de Conjuntos)](#8-álgebra-relacional-sql-como-teoria-de-conjuntos)
9. [Reciprocal Rank Fusion (RRF)](#9-reciprocal-rank-fusion-rrf)
10. [Métricas de Avaliação](#10-métricas-de-avaliação)
11. [Checklist final de domínio](#11-checklist-final-de-domínio)

---

## 1. Vetores e Embeddings — "significado vira geometria"

### Teoria
Um **vetor** é só uma lista ordenada de números: `[0.2, -0.5, 0.9, ...]`. Um **embedding** é o
vetor que um modelo (no PolyRAG, o **BGE-M3**) gera para representar o *significado* de um texto.
A propriedade mágica: textos com significados parecidos geram vetores que **apontam para direções
parecidas** no espaço — a proximidade geométrica passa a codificar proximidade semântica.

O BGE-M3 gera vetores de **1024 dimensões** e é multilíngue (cobre PT-BR e EN igualmente bem, o
que importa porque o projeto e os dados são em português).

### Analogia
Pense em cada palavra/frase como uma cidade num mapa gigante de milhares de dimensões (não só
latitude/longitude, mas 1024 "coordenadas" de significado). Cidades culturalmente parecidas
("gato" e "gatinho") ficam vizinhas; cidades sem relação ("gato" e "declaração de imposto de
renda") ficam em polos opostos do mapa. Buscar "textos parecidos" vira literalmente "achar as
cidades mais próximas no mapa".

### Exemplo prático
```python
import numpy as np

# Simulação: cada "embedding" real teria 1024 dimensões; aqui usamos 4 só para visualizar.
gato       = np.array([0.90, 0.10, 0.05, 0.02])
gatinho    = np.array([0.85, 0.15, 0.05, 0.03])   # parecido com "gato" -> vetor próximo
declaracao_ir = np.array([0.02, 0.03, 0.95, 0.80]) # sem relação -> vetor distante

print("gato -> gatinho :", np.linalg.norm(gato - gatinho))       # pequeno
print("gato -> declaracao_ir:", np.linalg.norm(gato - declaracao_ir))  # grande
```
Rodando isso você vê que a distância euclidiana entre `gato` e `gatinho` é bem menor do que entre
`gato` e `declaracao_ir` — a geometria reflete o significado.

### Onde é usado no PolyRAG
Roteador semântico (decide para qual RAG mandar o chunk), cache CAG (compara pergunta nova com
perguntas já respondidas), RAG 2/Qdrant (busca de textos por proximidade), RAG 3 (match de
entidades no grafo). Gerado sempre via `llama-server :8082 /v1/embeddings` — o projeto **nunca**
importa um modelo de embeddings direto em Python.

### Auto-teste
> Por que dois textos sinônimos, mas escritos com palavras totalmente diferentes ("carro" vs.
> "automóvel"), ainda produzem vetores próximos? (Resposta: o embedding captura significado, não
> a superfície lexical — o modelo foi treinado para isso.)

---

## 2. Norma L2 e Normalização — "padronizar o tamanho das flechas"

### Teoria
A **norma L2** de um vetor é o seu comprimento — generalização do Teorema de Pitágoras para N
dimensões:

```
‖v‖ = √(v₁² + v₂² + ... + vₙ²)
```

**Normalizar** um vetor significa dividi-lo pelo seu próprio comprimento:

```
normalize(v) = v / ‖v‖
```

O vetor resultante tem comprimento **exatamente 1** (vetor unitário) — sobra só a *direção*,
elimina-se o "tamanho". Isso importa porque, depois de normalizar, **similaridade de cosseno vira
produto escalar puro e simples** (ver tópico 3), que é muito mais barato de calcular. É o truque
usado antes de indexar qualquer vetor no FAISS `IndexFlatIP`.

### Analogia
Duas flechas apontando exatamente na mesma direção, mas uma tem 2 metros e a outra 20 centímetros
— para saber "para onde elas apontam" o tamanho é irrelevante e só atrapalha a comparação.
Normalizar é encolher (ou esticar) todas as flechas para o mesmo tamanho (1), sobrando só a
direção que interessa.

### Exemplo prático
```python
import numpy as np

v = np.array([3.0, 4.0])          # comprimento clássico 3-4-5
norma = np.linalg.norm(v)         # = 5.0
v_unit = v / norma

print("norma:", norma)                     # 5.0
print("vetor normalizado:", v_unit)        # [0.6, 0.8]
print("norma do vetor normalizado:", np.linalg.norm(v_unit))  # 1.0 (sempre!)
```

### Onde é usado no PolyRAG
Todo embedding é L2-normalizado **antes** de entrar no `faiss.IndexFlatIP` (CAG e índice do
roteador) — assim "inner product" (produto interno) equivale exatamente a cosseno, sem custo
extra de calcular normas toda hora durante a busca.

### Auto-teste
> Se um vetor já está normalizado (`‖v‖ = 1`), o que acontece se você normalizá-lo de novo?
> (Resposta: nada muda — dividir por 1 é a identidade. Normalizar é *idempotente*.)

---

## 3. Produto Escalar e Similaridade de Cosseno

### Teoria
O **produto escalar** (dot product) entre dois vetores é a soma dos produtos das componentes:

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

A **similaridade de cosseno** usa o produto escalar para medir o **ângulo** entre dois vetores,
ignorando o comprimento deles:

```
                a · b
cos(θ) = ─────────────────
           ‖a‖ × ‖b‖
```

- `cos = 1` → mesma direção (significado idêntico).
- `cos ≈ 0` → direções perpendiculares (sem relação).
- Na prática, com embeddings de texto, o resultado fica entre 0 e 1.

**Por que cosseno e não distância euclidiana?** Significado está na *direção*, não na magnitude —
um documento longo é um vetor "maior", mas o assunto dele é dado pela direção, não pelo tamanho.
Cosseno ignora magnitude por construção; euclidiana não.

Depois que os vetores estão normalizados (tópico 2), `cos(θ) = a · b` — a divisão pelas normas some
porque `‖a‖ = ‖b‖ = 1`. É por isso que o FAISS `IndexFlatIP` (inner product) calcula cosseno "de
graça".

### Analogia
Dois relógios analógicos: o que importa para saber "que horas são" é o **ângulo** dos ponteiros,
não o tamanho físico do relógio. Um relógio de parede gigante e um relógio de pulso minúsculo
marcando a mesma hora têm ponteiros apontando para o mesmo ângulo — cosseno = 1, mesmo os vetores
tendo "tamanhos" (normas) completamente diferentes.

### Exemplo prático
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pergunta        = np.array([0.9, 0.1, 0.3])
frase_ancora_A  = np.array([0.85, 0.15, 0.25])   # rota "financeiro"
frase_ancora_B  = np.array([-0.2, 0.9, -0.4])    # rota "suporte técnico"

print("cos(pergunta, A):", cosine_similarity(pergunta, frase_ancora_A))  # alto, ~0.98
print("cos(pergunta, B):", cosine_similarity(pergunta, frase_ancora_B))  # baixo/negativo
```

### Onde é usado no PolyRAG
Literalmente em todo o projeto: roteador (compara pergunta com frases-âncora de cada rota), CAG
(compara pergunta nova com perguntas já em cache, threshold `cosseno ≥ 0.92`), Qdrant/RAG 2
(distância configurada como `Cosine` no `config.yaml`), match de entidades no grafo/RAG 3.

### Auto-teste
> Se `cos(a, b) = -1`, o que isso significa geometricamente? (Resposta: vetores apontando em
> direções exatamente opostas — 180°. Raro em embeddings de texto reais, mas matematicamente
> válido.)

---

## 4. Margem de Confiança — o Roteador como Classificador 1-NN

### Teoria
O roteador semântico do PolyRAG embeda a pergunta/chunk de entrada, calcula o cosseno contra
**todas** as frases-âncora de **todas** as rotas (definidas no `config.yaml`), guarda o melhor
score por rota, e então calcula:

```
top1   = melhor score entre todas as rotas
top2   = segundo melhor score
margin = top1 − top2
```

A decisão segue uma regra de 3 zonas (valores reais do `config.yaml` do projeto):

```
top1 ≥ tau_high (0.62)  E  margin ≥ delta_margin (0.08)   → rota = top1  (confiante, custo ~0)
top1 < tau_low (0.35)                                      → fallback → vectorial (RAG 2)
caso contrário                                              → zona cinzenta → LLM-as-judge
```

Conceitualmente isso é um **classificador 1-NN (nearest neighbor) com regra de rejeição**: em vez
de sempre confiar na resposta mais próxima, o sistema tem uma zona onde admite "não tenho certeza
suficiente" e escala a decisão para um juiz mais caro (o LLM). É um padrão clássico de ML aplicado
de forma 100% determinística e testável (sem LLM envolvido no cálculo em si).

### Analogia
Pense num goleiro decidindo se sai do gol para interceptar a bola. Se a bola está claramente mais
perto dele do que de qualquer atacante (`top1` alto, `margin` grande) ele sai com confiança. Se a
bola está muito longe de todo mundo (`top1 < tau_low`), ele nem tenta — deixa para a defesa
(fallback). Se está numa distância ambígua, parecida entre ele e um atacante (`margin` pequena),
ele hesita e pede ajuda ao zagueiro mais experiente (LLM-as-judge).

### Exemplo prático
```python
import numpy as np

TAU_HIGH = 0.62
TAU_LOW = 0.35
DELTA_MARGIN = 0.08

def decidir_rota(scores_por_rota: dict) -> str:
    ranked = sorted(scores_por_rota.items(), key=lambda kv: kv[1], reverse=True)
    (rota1, top1), (rota2, top2) = ranked[0], ranked[1]
    margin = top1 - top2

    if top1 >= TAU_HIGH and margin >= DELTA_MARGIN:
        return f"rota={rota1} (confiante, top1={top1:.2f}, margin={margin:.2f})"
    if top1 < TAU_LOW:
        return "fallback -> vectorial (nenhuma rota confiante)"
    return f"zona cinzenta -> LLM-as-judge (top1={top1:.2f}, margin={margin:.2f})"

# Caso 1: confiante
print(decidir_rota({"financeiro": 0.81, "suporte": 0.40, "juridico": 0.20}))
# Caso 2: ambíguo (top1 alto mas margem pequena)
print(decidir_rota({"financeiro": 0.70, "juridico": 0.68, "suporte": 0.10}))
# Caso 3: fallback
print(decidir_rota({"financeiro": 0.30, "suporte": 0.20, "juridico": 0.15}))
```

### Onde é usado no PolyRAG
Roteador de ingestão (Dia 2 do roadmap) e roteador de busca (Dia 4, mesma engine reaproveitada).
Cada decisão é logada num span do OpenTelemetry (`router.score_top1`, `router.score_top2`,
`router.margin`, `router.decision_stage`) — auditável, sem "caixa-preta".

### Auto-teste
> Por que a regra usa `margin` além de `top1`? Por que não bastaria checar só `top1 ≥ tau_high`?
> (Resposta: `top1` alto sozinho não garante que a rota está certa — se `top2` está quase igual a
> `top1`, a decisão é ambígua mesmo com `top1` "confiante". A margem detecta empate técnico.)

---

## 5. Busca Vetorial: Força Bruta (FAISS) vs. Aproximada (HNSW/Qdrant)

### Teoria
**Busca exata — `faiss.IndexFlatIP`:** compara a query contra **todos** os N vetores, um a um, via
produto interno (que já é cosseno, porque os vetores estão normalizados). Parece ingênuo, mas com
N pequeno (o cache CAG do projeto guarda até ~10 mil vetores) o resultado sai em **microssegundos**
porque o FAISS usa instruções SIMD da CPU (várias multiplicações por ciclo de clock). É **exata**
— não é uma aproximação, sempre acha o melhor vizinho de verdade.

**Busca aproximada — HNSW (dentro do Qdrant):** quando N chega a milhões, força bruta não escala
(seria O(N) por busca). HNSW constrói um **grafo em múltiplas camadas navegáveis**: a camada do
topo tem poucos nós com "atalhos" de longo alcance; as camadas inferiores têm ligações mais finas.
A busca desce de camada em camada, refinando — o custo vira aproximadamente O(log N). É
**aproximada**: troca uma fração de exatidão (recall ~99%) por velocidade.

### Analogia
Força bruta é como conferir o preço de um produto perguntando pessoalmente a cada uma das 10 mil
lojas de uma cidade — viável se são só 10 mil lojas e você é rapidíssimo (SIMD). HNSW é como usar
o sistema de rodovias: primeiro pega a rodovia (camada de longo alcance) até a região certa da
cidade, depois a avenida, depois a rua — chega perto da resposta certa em poucos "saltos", mas não
literalmente conferiu *todas* as lojas, então existe uma chance mínima de ter passado batido pela
melhor.

### Exemplo prático
```python
import numpy as np

def busca_forca_bruta(query, vetores, k=3):
    """O(N): compara a query contra TODOS os vetores. Exato."""
    scores = vetores @ query          # produto interno com cada vetor (já normalizados)
    top_k_idx = np.argsort(-scores)[:k]
    return list(zip(top_k_idx, scores[top_k_idx]))

np.random.seed(0)
N, DIM = 2000, 8
vetores = np.random.randn(N, DIM)
vetores /= np.linalg.norm(vetores, axis=1, keepdims=True)  # normaliza todos (tópico 2)

query = vetores[42] + np.random.randn(DIM) * 0.01           # query "parecida" com o vetor 42
query /= np.linalg.norm(query)

print(busca_forca_bruta(query, vetores, k=3))
# O vetor de índice 42 deve aparecer no topo, com score próximo de 1.0
```
(HNSW real é implementado dentro do Qdrant — reimplementar do zero foge do escopo de estudo, mas o
essencial para entrevista é entender o trade-off exato-vs-aproximado e por que cada um foi
escolhido para cada caso de uso.)

### Onde é usado no PolyRAG
`IndexFlatIP` no CAG (cache) e no índice do roteador — poucos milhares de vetores, precisa ser
exato e instantâneo. HNSW dentro do Qdrant para o RAG 2 (memória de textos) — potencialmente
milhões de chunks, prioriza escala sobre exatidão perfeita.

### Auto-teste
> Por que faria sentido usar HNSW no roteador em vez de força bruta? (Resposta: normalmente não
> faz — o roteador tem poucas dezenas/centenas de frases-âncora, então força bruta já é
> instantânea e 100% exata; HNSW só compensa quando N é grande o suficiente para o overhead do
> grafo valer a pena.)

---

## 6. Grafos, PageRank e Personalized PageRank

### Teoria
Um **grafo** é um conjunto de **nós** (no RAG 3: entidades como "Fornecedor A", "Lei 13.709") e
**arestas dirigidas** (relações entre elas, extraídas via NER).

**PageRank clássico** (algoritmo original do Google, 1998): a importância de um nó vem da
importância de quem aponta para ele. Modelado como um **passeio aleatório**: um "turista" está num
nó e, a cada passo, segue uma aresta de saída escolhida aleatoriamente. A fração de tempo que ele
passa em cada nó, no longuíssimo prazo, é o PageRank daquele nó.

```
PR(n) = (1−d)/N + d × Σ PR(vizinho) / grau_saída(vizinho)         d = 0.85 (fator de amortecimento)
```

Matematicamente, isso é o **autovetor principal da matriz de transição** — a distribuição
estacionária de uma cadeia de Markov. O `networkx` resolve isso por iteração (repetir a fórmula
~100 vezes até convergir).

**Personalized PageRank (PPR)** — usado no RAG 3: o termo de teletransporte `(1−d)/N` deixa de
voltar para *qualquer* nó e passa a voltar **só para os nós-semente** (as entidades detectadas na
pergunta do usuário). Resultado: o ranking passa a medir *"relevância em relação a ESTA
pergunta"*, não importância global do grafo inteiro.

**Por que isso dá raciocínio multi-hop:** a "importância" escorre pelas arestas por vários saltos.
Se a pergunta cita "Fornecedor A", o PPR espalha crédito para "Contrato Y" (que depende de A) e
dali para "Cláusula Z" — dois saltos de raciocínio emergindo puramente da propagação no grafo, sem
nenhuma regra `if` programada.

> ⚠️ **Armadilha real do projeto:** versões recentes do `networkx` (3.6+) **removeram**
> `nx.personalized_pagerank` e passaram a exigir `scipy` para `nx.pagerank`. A skill do projeto
> documenta isso e recomenda implementar a **iteração de potência (power iteration) em NumPy
> puro** — validada para bater com o `networkx` com erro < 1e-4, e deterministicamente reprodutível
> bit a bit.

### Analogia
PageRank clássico: numa festa, a pessoa mais "importante" é aquela que mais gente menciona/aponta
como referência — e se as pessoas que a mencionam também são muito mencionadas, o efeito é ainda
mais forte (é recursivo). Personalized PageRank é a mesma festa, mas perguntando "quem é
importante *em relação ao meu grupo de amigos específico*" — o cálculo sempre volta para as pessoas
do seu grupo (nós-semente), não para qualquer um da festa.

### Exemplo prático (power iteration pura em NumPy — a versão "à prova de breaking change")
```python
import numpy as np

def personalized_pagerank(adj: np.ndarray, seeds: list[int], d: float = 0.85, tol: float = 1e-6,
                            max_iter: int = 100) -> np.ndarray:
    """
    adj: matriz de adjacência N x N (adj[i, j] = 1 se existe aresta i -> j)
    seeds: índices dos nós-semente (entidades detectadas na pergunta)
    """
    n = adj.shape[0]
    out_degree = adj.sum(axis=1)
    dangling = (out_degree == 0)                      # nós sem nenhuma aresta de saída

    # vetor de personalização: toda a "massa" de teletransporte volta só para os seeds
    personalization = np.zeros(n)
    personalization[seeds] = 1.0 / len(seeds)

    # matriz de transição normalizada (evita divisão por zero em nós dangling)
    out_degree_safe = np.where(out_degree == 0, 1, out_degree)
    transition = adj / out_degree_safe[:, None]

    r = personalization.copy()
    for _ in range(max_iter):
        dangling_mass = r[dangling].sum()               # massa "perdida" nos nós sem saída
        r_new = (1 - d) * personalization
        r_new += d * (transition.T @ r)                  # propaga pelos vizinhos de entrada
        r_new += d * dangling_mass * personalization      # redistribui massa perdida
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r

# Grafo de exemplo: 0="Fornecedor A", 1="Contrato Y", 2="Cláusula Z", 3="Fornecedor B" (isolado)
adj = np.array([
    [0, 1, 0, 0],   # Fornecedor A -> Contrato Y
    [0, 0, 1, 0],   # Contrato Y -> Cláusula Z
    [0, 0, 0, 0],   # Cláusula Z (sem saída, "dangling")
    [0, 0, 0, 0],   # Fornecedor B (isolado)
])

scores = personalized_pagerank(adj, seeds=[0])   # pergunta menciona só "Fornecedor A"
for i, s in enumerate(scores):
    print(f"nó {i}: {s:.4f}")
# Esperado: nó 0 (semente) alto, nó 1 (1 salto) médio, nó 2 (2 saltos) menor, nó 3 (isolado) ~0
```

### Onde é usado no PolyRAG
RAG 3 inteiro (Dia 3): NER extrai entidades da pergunta → viram nós-semente → PPR ranqueia chunks
do grafo por relevância à pergunta → chunks mais relevantes viram contexto para o LLM.

### Auto-teste
> Se você aumentar `d` (fator de amortecimento) de 0.85 para 0.99, o PageRank fica mais ou menos
> "espalhado" pelo grafo? (Resposta: mais espalhado/propagado — `d` alto dá mais peso ao passeio
> pelas arestas e menos ao teletransporte de volta às sementes, favorecendo nós mais distantes.)

---

## 7. Quantização de Modelos

### Teoria
Os pesos (parâmetros) de um modelo de IA normalmente são armazenados como números de ponto
flutuante de 16 bits (FP16). **Quantização** é o processo de comprimir esses números para uma
representação mais compacta — no projeto, ~4 bits (formato `Q4_K_M`) — trocando um pouco de
precisão numérica por uma redução de ~4x na memória usada, com perda mínima de qualidade prática.

```
Qwen3-14B em FP16:  14 bilhões de parâmetros × 2 bytes ≈ 28 GB   → não cabe em 16GB de VRAM
Qwen3-14B em Q4:    14 bilhões de parâmetros × 0.5 byte ≈ 9 GB   → cabe confortavelmente
```

### Analogia
É como salvar uma foto em alta resolução (RAW, arquivo gigante) versus a mesma foto em JPEG
comprimido: o JPEG ocupa uma fração do espaço e, para o olho humano, a diferença de qualidade é
quase imperceptível na maioria dos usos — só em ampliação extrema (zoom absurdo) você nota a
perda. Quantização faz o mesmo com os números do modelo: reduz a "resolução numérica" dos pesos,
mas para a tarefa de gerar texto a perda é pequena o bastante para não importar na prática.

### Exemplo prático (ilustração conceitual do que "reduzir bits" significa)
```python
import numpy as np

def quantizar_ilustrativo(pesos_fp32: np.ndarray, n_niveis: int = 16) -> np.ndarray:
    """
    Ilustra o princípio (não é o algoritmo real do llama.cpp, que usa blocos e escala/zero-point
    por bloco): mapeia valores contínuos para um número finito de "baldes" (níveis).
    n_niveis=16 é aproximadamente o que 4 bits permitem (2^4 = 16 valores possíveis).
    """
    minimo, maximo = pesos_fp32.min(), pesos_fp32.max()
    passo = (maximo - minimo) / (n_niveis - 1)
    indices = np.round((pesos_fp32 - minimo) / passo).astype(int)
    return minimo + indices * passo   # valores "arredondados" para os níveis permitidos

pesos_originais = np.array([0.123, -0.876, 0.451, 0.998, -0.002, 0.333])
pesos_quantizados = quantizar_ilustrativo(pesos_originais)

print("originais:   ", pesos_originais)
print("quantizados: ", pesos_quantizados)
print("erro médio:  ", np.abs(pesos_originais - pesos_quantizados).mean())
```

### Onde é usado no PolyRAG
Os 3 modelos servidos via `llama.cpp` (Qwen3-14B para respostas, GLM-OCR para extração de texto de
imagens, BGE-M3 para embeddings) rodam quantizados — é o que permite os três caberem juntos em
16GB de VRAM numa GPU AMD via Vulkan, sem PyTorch/Docker/WSL.

### Auto-teste
> Por que reduzir de FP16 para Q4 não é simplesmente "cortar metade dos bits" ingenuamente?
> (Resposta: implementações reais como `Q4_K_M` usam escala e zero-point por bloco de pesos para
> minimizar o erro de arredondamento — não é um corte uniforme, é uma compressão com calibração.)

---

## 8. Álgebra Relacional (SQL como Teoria de Conjuntos)

### Teoria
SQL não é "mágica de banco de dados" — é teoria de conjuntos aplicada a tabelas:

| Cláusula SQL | Operação de álgebra relacional | Símbolo |
|---|---|---|
| `WHERE` | Seleção (filtra linhas por condição) | σ (sigma) |
| `SELECT colunas` | Projeção (filtra colunas) | π (pi) |
| `JOIN` | Produto cartesiano filtrado (combina tabelas, depois filtra pares válidos) | ⨝ |
| `GROUP BY` | Partição do conjunto + agregação por partição | — |

Entender isso ajuda a explicar/depurar queries geradas por Text-to-SQL: toda query complexa é só
composição dessas operações básicas.

### Analogia
Pense numa planilha gigante. `WHERE` é "esconder as linhas que não interessam" (peneira
horizontal). `SELECT colunas` é "esconder as colunas que não interessam" (peneira vertical).
`JOIN` é "colar duas planilhas lado a lado combinando só as linhas que compartilham uma chave em
comum" (como um Procv/VLOOKUP generalizado). `GROUP BY` é "separar a planilha em pilhas menores por
categoria e resumir cada pilha num número só" (como uma tabela dinâmica).

### Exemplo prático
```python
# Ilustração em Python puro do que o SQL faz "por baixo dos panos" com listas de dicts
vendas = [
    {"produto": "A", "regiao": "SP", "valor": 100},
    {"produto": "B", "regiao": "SP", "valor": 200},
    {"produto": "A", "regiao": "RJ", "valor": 150},
]

# WHERE regiao = 'SP'  -> seleção (σ)
selecao = [v for v in vendas if v["regiao"] == "SP"]

# SELECT produto, valor -> projeção (π)
projecao = [{"produto": v["produto"], "valor": v["valor"]} for v in selecao]

# GROUP BY produto, SUM(valor) -> partição + agregação
from collections import defaultdict
agrupado = defaultdict(int)
for v in vendas:
    agrupado[v["produto"]] += v["valor"]

print("seleção (WHERE):", selecao)
print("projeção (SELECT):", projecao)
print("agregação (GROUP BY):", dict(agrupado))
```

### Onde é usado no PolyRAG
RAG 1 (Text-to-SQL sobre SQLite): o LLM traduz a pergunta em linguagem natural para SQL, que é
validado como **read-only** antes de executar contra a base relacional.

### Auto-teste
> Por que `JOIN` é descrito como "produto cartesiano filtrado" e não só "combinar tabelas"?
> (Resposta: conceitualmente o banco primeiro gera todas as combinações possíveis de linhas das
> duas tabelas — produto cartesiano — e só depois mantém as combinações que satisfazem a condição
> do `ON`; na prática o otimizador do banco nunca materializa isso literalmente, mas é o modelo
> matemático correto.)

---

## 9. Reciprocal Rank Fusion (RRF)

### Teoria
Quando uma pergunta é **composta** e o roteador decide consultar mais de um RAG ao mesmo tempo
(fan-out), cada RAG retorna uma lista de chunks ranqueados — mas em **escalas diferentes**: SQL
retorna um valor agregado, Qdrant retorna cosseno (0 a 1), o grafo retorna PageRank. Somar esses
scores diretamente seria comparar grandezas incompatíveis.

RRF resolve isso ignorando os valores de score e usando **só a posição no ranking** (rank):

```
score_final(chunk) = Σ_rotas  1 / (k + rank_na_rota)          k = 60 (constante clássica da literatura de IR)
```

Exemplo: um chunk que ficou em 2º lugar no SQL e 5º lugar no grafo:
```
1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151
```

Chunks que aparecem bem ranqueados em **múltiplas** rotas acabam com score final mais alto —
naturalmente "premiando" o consenso entre fontes diferentes.

### Analogia
É como decidir o vencedor de uma competição avaliada por três jurados que usam escalas totalmente
diferentes (um dá nota de 0 a 10, outro de 0 a 100, outro só "ouro/prata/bronze"). Somar as notas
brutas não faria sentido. Em vez disso, você olha só a **posição** que cada jurado deu a cada
competidor (1º, 2º, 3º lugar) e combina as posições — isso funciona não importa a escala de nota
que cada jurado usou.

### Exemplo prático
```python
def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """
    rankings: {"sql": ["chunk_9", "chunk_2", ...], "qdrant": [...], "grafo": [...]}
    Cada lista já vem ordenada da mais relevante para a menos relevante NAQUELA rota.
    """
    scores: dict[str, float] = {}
    for rota, chunks_ranqueados in rankings.items():
        for rank_zero_based, chunk_id in enumerate(chunks_ranqueados):
            rank = rank_zero_based + 1                 # rank começa em 1, não em 0
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

rankings = {
    "sql":    ["chunk_A", "chunk_C", "chunk_B"],
    "qdrant": ["chunk_B", "chunk_A", "chunk_D"],
    "grafo":  ["chunk_A", "chunk_D", "chunk_C"],
}

for chunk_id, score in reciprocal_rank_fusion(rankings).items():
    print(f"{chunk_id}: {score:.5f}")
# chunk_A aparece bem ranqueado nas 3 rotas -> deve vencer no score final
```

### Onde é usado no PolyRAG
Fan-out de perguntas compostas (Dia 4, caminho de busca): quando o roteador identifica que a
pergunta precisa de mais de uma base, cada RAG retorna sua lista, e o RRF funde tudo numa lista
final única, sem precisar normalizar escalas heterogêneas.

### Auto-teste
> Por que RRF usa `k=60` em vez de `k=0` na fórmula `1/(k+rank)`? (Resposta: `k` suaviza a
> diferença de peso entre os primeiros lugares — sem ele, o 1º lugar teria peso `1/1=1.0` e o 2º
> `1/2=0.5`, uma queda abrupta de 50%; com `k=60`, a queda entre 1º e 2º é muito mais suave,
> evitando que uma única rota domine o resultado só por ter colocado um chunk em 1º lugar.)

---

## 10. Métricas de Avaliação

### Teoria
Avaliar um sistema de RAG exige métricas em duas camadas distintas:

- **Recall@k**: dos itens relevantes que existem, quantos foram recuperados dentro dos top-k
  resultados? Mede se a informação certa **chegou** ao contexto do LLM.
- **Precision@k**: dos k itens recuperados, quantos são de fato relevantes? Mede quanto "lixo"
  irrelevante entrou no contexto junto com o que importa.
- **router_accuracy**: fração das vezes em que o roteador mandou o chunk/pergunta para a rota
  correta (comparado contra um gabarito/golden set).
- **answer_correctness**: a resposta final gerada pelo LLM está correta? Normalmente avaliada via
  LLM-as-judge com uma rubrica binária (certo/errado) contra um golden set de perguntas e respostas
  esperadas.

A distinção crucial: se `recall@k` está alto mas `answer_correctness` está baixo, o problema está
na **geração** (o LLM tinha a informação certa e respondeu errado mesmo assim). Se `recall@k` já
está baixo, o problema está na **recuperação** (a informação nem chegou ao LLM) — não adianta
trocar de LLM, o problema é upstream.

### Analogia
Pense num aluno fazendo uma prova de consulta. `Recall@k` é "quantas das páginas do livro que
tinham a resposta certa ele conseguiu abrir a tempo". `Precision@k` é "das páginas que ele abriu,
quantas eram realmente úteis (e não só ruído)". `Answer_correctness` é a nota final da prova — e
pode ser baixa mesmo que ele tenha aberto a página certa, se ele leu errado ou escreveu a resposta
errada mesmo com a informação certa na frente dele.

### Exemplo prático
```python
def recall_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    topo_k = set(recuperados[:k])
    if not relevantes:
        return 0.0
    return len(topo_k & relevantes) / len(relevantes)

def precision_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    topo_k = recuperados[:k]
    if not topo_k:
        return 0.0
    acertos = sum(1 for item in topo_k if item in relevantes)
    return acertos / len(topo_k)

recuperados = ["chunk_9", "chunk_2", "chunk_5", "chunk_1", "chunk_7"]
relevantes = {"chunk_2", "chunk_1", "chunk_4"}   # gabarito: 3 chunks realmente relevantes

print("recall@5:   ", recall_at_k(recuperados, relevantes, k=5))     # 2 de 3 relevantes -> 0.667
print("precision@5:", precision_at_k(recuperados, relevantes, k=5))  # 2 de 5 recuperados -> 0.4

def router_accuracy(previstas: list[str], gabarito: list[str]) -> float:
    acertos = sum(1 for p, g in zip(previstas, gabarito) if p == g)
    return acertos / len(gabarito)

previstas = ["financeiro", "juridico", "suporte", "financeiro"]
gabarito  = ["financeiro", "juridico", "juridico", "financeiro"]
print("router_accuracy:", router_accuracy(previstas, gabarito))  # 3 de 4 -> 0.75
```

### Onde é usado no PolyRAG
Scripts de avaliação da skill `polyrag-dev` (`evals/evals.json`), usados para validar cada RAG
isoladamente (recall@k, precision@k) e o roteador (router_accuracy) contra golden sets, além de
answer_correctness fim-a-fim via LLM-as-judge.

### Auto-teste
> Se `precision@10` está baixa mas `precision@3` está alta, o que isso sugere sobre o ranking dos
> resultados? (Resposta: os itens mais relevantes estão bem ranqueados no topo, mas a lista "engorda"
> com ruído conforme desce — o problema não é a qualidade do que existe no índice, é onde ele fica
> posicionado no ranking.)

---

## 11. Checklist final de domínio

Marque quando conseguir responder **sem consultar** este documento:

- [ ] Explicar, com uma frase, por que "significado vira geometria" em embeddings.
- [ ] Calcular na mão a norma L2 de um vetor pequeno (ex.: `[3, 4]`) e normalizá-lo.
- [ ] Explicar por que, após normalizar, cosseno = produto escalar.
- [ ] Dizer de cabeça os 3 thresholds do roteador (`tau_high`, `tau_low`, `delta_margin`) e o que
      cada zona de decisão significa.
- [ ] Explicar a diferença entre busca exata (FAISS `IndexFlatIP`) e aproximada (HNSW) e quando
      cada uma é a escolha certa.
- [ ] Descrever PageRank como um passeio aleatório e explicar a diferença para Personalized
      PageRank em uma frase ("teletransporte volta só para os seeds").
- [ ] Justificar por que a versão do power iteration em NumPy puro existe no projeto (breaking
      change do networkx 3.6+).
- [ ] Explicar quantização sem usar a palavra "comprime" (ex.: "reduz a precisão numérica dos
      pesos para caber em menos memória").
- [ ] Mapear `WHERE`/`SELECT`/`JOIN`/`GROUP BY` para seleção/projeção/produto-cartesiano-filtrado/
      partição+agregação.
- [ ] Explicar por que RRF usa apenas ranks, não os scores brutos.
- [ ] Diferenciar, com um exemplo, um problema de recall (retrieval) de um problema de
      answer_correctness (geração).

---

*Documento gerado para apoiar o estudo teórico do PolyRAG — os valores numéricos e nomes de
função referenciados (`tau_high`, `IndexFlatIP`, `personalized_pagerank`, etc.) correspondem à
implementação real descrita em `CLAUDE.md` e na skill `polyrag-dev` do projeto.*
