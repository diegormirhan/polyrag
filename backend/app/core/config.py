from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

# Sobe 3 níveis a partir deste arquivo (core/ -> app/ -> backend/ -> raiz do projeto)
# para achar o config.yaml na raiz, sem depender de qual diretório o processo foi iniciado.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

class ServerConfig(BaseModel):
    host: str
    port: int

# Cada llama-server roda um modelo diferente (LLM, OCR, embeddings), então cada um
# tem seu próprio bloco de config com host/porta/ctx_size independentes.
class LlmModelConfig(BaseModel):
    model_path: Path
    host: str
    port: int
    ctx_size: int
    n_gpu_layers: int
    temperature: float = 0.2  # baixa de propósito: determinismo (ver seção 0.1 do CLAUDE.md)
    # -1 = nunca dorme. > 0 faz o llama-server devolver a VRAM depois de N segundos
    # ocioso e recarregar na próxima requisição — medido: o GLM-OCR devolve 2.08GB.
    sleep_idle_seconds: int = -1

class OcrModelConfig(BaseModel):
    model_path: Path
    mmproj_path: Path  # projeção multimodal (imagem -> espaço do modelo) exigida por modelos de visão no llama.cpp
    host: str
    port: int
    ctx_size: int
    n_gpu_layers: int
    sleep_idle_seconds: int = -1  # ver LlmModelConfig

class EmbeddingsModelsConfig(BaseModel):
    model_path: Path
    host: str
    port: int
    ctx_size: int
    n_gpu_layers: int
    dimensions: int  # tamanho do vetor de saída (1024 pro BGE-M3) — usado pelo Qdrant e pelo FAISS

class LlamaConfig(BaseModel):
    bin_path: Path  # caminho do executável llama-server.exe, compartilhado pelas 4 instâncias
    llm: LlmModelConfig
    ocr: OcrModelConfig
    embeddings: EmbeddingsModelsConfig
    judge: LlmModelConfig  # Prometheus 2 — juiz dedicado do roteador (mesmo shape do llm, sem mmproj)

class QdrantConfig(BaseModel):
    bin_path: Path
    storage_path: Path  # onde o Qdrant persiste as coleções (passado via env na subida)
    host: str
    port: int
    collection: str
    distance: str
    vector_size: int

class PathsConfig(BaseModel):
    data_drop: str      # hot folder onde o usuário larga arquivos
    processed: str       # arquivos já ingeridos são movidos pra cá
    sqlite_db: str        # RAG 1
    graph_store: str      # RAG 3 (grafo persistido em JSON)

# Prompts de cada RAG ficam aqui (mesma regra do ocr_prompt/llm_judge_prompt:
# nada de prompt hardcoded no código). RAG 2 e RAG 3 entram nesta seção conforme
# forem precisando dos seus próprios prompts.
class RelationalRagConfig(BaseModel):
    text_to_sql_prompt: str  # template com {schema} e {question}

class GraphRagConfig(BaseModel):
    entity_match_threshold: float  # cosseno mínimo pra casar entidade da pergunta com nó do grafo
    pagerank_damping: float        # fator d do PageRank (ver seção 4.6 do CLAUDE.md)
    openie_prompt: str             # ingestão: {text} -> triplas em JSON (texto declarativo)
    ner_prompt: str                # busca: {text} -> lista de entidades (pergunta não declara fato)

class RagsConfig(BaseModel):
    relational: RelationalRagConfig
    graph: GraphRagConfig

# Cada rota do roteador semântico carrega uma descrição (usada pelo LLM-juiz na zona
# cinzenta) e uma lista de frases-âncora (usadas para gerar os vetores de referência
# contra os quais o cosseno é calculado — é assim que o roteador é "treinado" via YAML).
class RouteConfig(BaseModel):
    description: str
    utterances: list[str]

class RouterConfig(BaseModel):
    tau_heuristic: float  # score mínimo do Estágio 1 (tabularity_score) pra decidir "relational" sem embedding
    tau_high: float       # score mínimo do top1 pra aceitar a rota direto
    tau_low: float          # abaixo disso, nem tenta: cai no fallback vectorial
    delta_margin: float      # margem mínima (top1 - top2) pra considerar a decisão "confiante"
    llm_judge_enabled: bool   # permite desligar o juiz LLM e forçar decisão só por threshold
    llm_judge_prompt: str
    routes: dict[str, RouteConfig]  # chave = nome da rota ("relational", "vectorial", "graph")

class CacheConfig(BaseModel):
    enabled: bool
    similarity_threshold: float  # cosseno mínimo pra considerar cache hit (CAG)
    max_entries: int

class OrchestratorConfig(BaseModel):
    top_k: int          # quantos resultados cada RAG devolve pro contexto do LLM
    answer_prompt: str  # template com {context} e {question}

class IngestConfig(BaseModel):
    watch_interval_s: float
    semantic_threshold: float  # cosseno mínimo entre frases consecutivas pra continuar no mesmo chunk
    min_chunk_chars: int  # piso de tamanho: abaixo disso, força continuar no mesmo grupo mesmo com cosseno baixo
    ocr_prompt: str
    vision_extensions: list[str]   # vão pro GLM-OCR antes de qualquer outra coisa
    text_extensions: list[str]     # lidos como string crua
    table_extensions: list[str]    # lidos como DataFrame (pandas)
    document_extensions: list[str]

class TelemetryConfig(BaseModel):
    enabled: bool
    service_name: str
    ring_buffer_size: int  # quantos spans recentes ficam guardados em RAM pro endpoint /telemetry/traces

# Modelo raiz: espelha 1:1 as chaves de topo do config.yaml. Se o YAML tiver um campo
# faltando ou de tipo errado, o Pydantic explode aqui na inicialização — não em algum
# ponto aleatório do pipeline em runtime.
class Settings(BaseModel):
    server: ServerConfig
    llama: LlamaConfig
    qdrant: QdrantConfig
    paths: PathsConfig
    rags: RagsConfig
    router: RouterConfig
    cache: CacheConfig
    orchestrator: OrchestratorConfig
    ingest: IngestConfig
    telemetry: TelemetryConfig

# lru_cache com um único argumento de default fixo funciona como singleton: a primeira
# chamada lê e valida o YAML; todas as chamadas seguintes (de qualquer módulo do backend)
# reaproveitam o mesmo objeto Settings sem reler o arquivo do disco.
@lru_cache(maxsize=1)
def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(**raw)
