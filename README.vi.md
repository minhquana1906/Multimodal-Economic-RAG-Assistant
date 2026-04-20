# Multimodal Economic RAG Assistant

> 🇻🇳 **Tiếng Việt** | [🇬🇧 English](README.md)

Hệ thống Retrieval-Augmented Generation (RAG) tương thích OpenAI, phục vụ tra cứu kinh tế và tài chính bằng tiếng Việt.

Stack kết hợp một **FastAPI orchestrator** (luồng LangGraph) với một **inference service** gộp (embedding + reranking trên GPU), cơ sở dữ liệu vector **Qdrant**, backend LLM **vLLM**, và giao diện chat **Open WebUI**.

---

## Mục lục

1. [Dự án làm gì](#1-dự-án-làm-gì)
2. [Kiến trúc tổng quan](#2-kiến-trúc-tổng-quan)
3. [State Machine LangGraph](#3-state-machine-langgraph)
4. [Chi tiết pipeline RAG](#4-chi-tiết-pipeline-rag)
5. [Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
6. [Các dịch vụ runtime](#6-các-dịch-vụ-runtime)
7. [Khái niệm & Lựa chọn thiết kế](#7-khái-niệm--lựa-chọn-thiết-kế)
8. [Tham khảo cấu hình](#8-tham-khảo-cấu-hình)
9. [Cài đặt & Khởi động nhanh](#9-cài-đặt--khởi-động-nhanh)
10. [Quy trình phát triển](#10-quy-trình-phát-triển)
11. [Kiểm thử](#11-kiểm-thử)
12. [Xử lý sự cố](#12-xử-lý-sự-cố)

---

## 1. Dự án làm gì

Người dùng đặt câu hỏi về kinh tế hoặc tài chính bằng tiếng Việt. Hệ thống truy xuất các đoạn văn bản liên quan từ kho tin tức kinh tế nội bộ, tuỳ chọn bổ sung kết quả tìm kiếm web, sau đó tạo câu trả lời có căn cứ với trích dẫn nội tuyến (`[S1]`, `[S2]` → liên kết có thể click).

**Khả năng chính:**

- Truy xuất hybrid dense+sparse với Reciprocal Rank Fusion (RRF)
- Fallback tìm kiếm web thông minh khi độ tin cậy thấp hoặc câu hỏi thời sự
- Tạo trích dẫn nội tuyến tự động, hậu xử lý thành markdown có thể click
- Định tuyến theo ý định — LLM quyết định mỗi câu hỏi dùng RAG hay trả lời trực tiếp
- Endpoint `/v1/chat/completions` tương thích OpenAI (streaming và non-streaming)

---

## 2. Kiến trúc tổng quan

```text
┌────────────────────────────────────────────────────────────────────────┐
│                       Người dùng / Open WebUI                          │
│                      POST /v1/chat/completions                         │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Orchestrator    │  :8000  FastAPI + LangGraph
                    │  (Định tuyến ý)   │
                    └──┬──────┬─────────┘
                       │      │
          RAG path     │      │  Direct path
                       │      │
          ┌────────────▼──┐  ┌▼────────────────┐
          │  RAG Graph    │  │  Trả lời trực tiếp│
          │  (LangGraph)  │  │  (± web context) │
          └──┬──────┬─────┘  └─────────────────┘
    embed    │      │ retrieve
             │      │
    ┌────────▼──┐  ┌▼────────────┐  ┌──────────────┐
    │ Inference │  │   Qdrant    │  │ Tavily (web) │
    │  :8001    │  │  :6333      │  │  (tuỳ chọn)  │
    │  embed    │  │  hybrid     │  │  fallback    │
    │  sparse   │  │  search     │  └──────────────┘
    │  rerank   │  └─────────────┘
    └───────────┘
             │
    ┌────────▼──────────┐
    │  LLM (vLLM)       │
    │  :8004            │
    │  Qwen3-4B-AWQ     │
    └───────────────────┘
```

**Luồng xử lý request:**

1. Orchestrator nhận request chat → gọi `detect_intent()` (LLM định tuyến JSON).
2. **Direct path:** LLM trả lời ngay, tuỳ chọn kèm context web.
3. **RAG path:** LangGraph chạy embed → retrieve → rerank → web fallback → combine → generate → citations.
4. Response được stream hoặc trả về một lần.

---

## 3. State Machine LangGraph

Toàn bộ pipeline RAG được mã hoá thành **`StateGraph` LangGraph** trên `RAGState`.

### Schema trạng thái

```text
RAGState
├── query             str           câu hỏi đã chuẩn hoá
├── raw_query         str           input gốc của người dùng
├── resolved_query    str           câu hỏi sau khi giải quyết ý định
├── task_type         str           "rag" | "direct" | "direct_web"
├── embeddings        list[float]   vector dense của câu hỏi
├── retrieved_docs    list[dict]    kết quả hybrid search (top_k=20)
├── reranked_docs     list[dict]    kết quả sau rerank (top_n=5)
├── web_results       list[dict]    kết quả Tavily web search
├── final_context     list[dict]    merge reranked + web
├── answer            str           câu trả lời được tạo
├── generation_prompt str           prompt gửi đến LLM
├── citations         list[dict]    metadata trích dẫn đã chuẩn hoá
├── citation_pool     dict          context_id → metadata đầy đủ
└── error             str | None    thông báo lỗi nếu node thất bại
```

### Topology đồ thị

```text
                        START
                          │
                   ┌──────▼──────┐
                   │  embed_node  │  vector dense qua InferenceClient
                   └──────┬──────┘
                          │ lỗi? → END
                   ┌──────▼──────┐
                   │retrieve_node│  hybrid_search (dense + sparse RRF)
                   └──────┬──────┘
                   ┌──────▼──────┐
                   │ rerank_node │  bge-reranker-v2-m3, giữ top_n kết quả
                   └──────┬──────┘
                   ┌──────▼──────────┐
                   │web_fallback_node│  kiểm tra policy → Tavily nếu kích hoạt
                   └──────┬──────────┘
                   ┌──────▼──────────────┐
                   │combine_context_node │  gộp docs + xây citation_pool
                   └──────┬──────────────┘
                          │  retrieval_only=True → END (dùng cho streaming)
                   ┌──────▼──────┐
                   │generate_node│  LLMClient.generate() với prompt [S1][S2]
                   └──────┬──────┘
                   ┌──────▼───────────┐
                   │  citations_node  │  viết lại [Sn] → [[Sn]](url) markdown
                   └──────┬───────────┘
                          END
```

**Chế độ streaming:** Đồ thị được gọi với `retrieval_only=True` để xây context, sau đó generation được stream trực tiếp qua `LLMClient.stream_chat()` bên ngoài đồ thị.

---

## 4. Chi tiết pipeline RAG

### 4.1 Truy xuất Hybrid (RRF)

Hai đường tìm kiếm song song được hợp nhất:

| Đường | Mô hình | Loại index |
|-------|---------|-----------|
| Dense | BAAI/bge-m3 (1024 chiều) | cosine HNSW |
| Sparse | Trọng số lexical BAAI/bge-m3 | Qdrant sparse |

Kết quả được kết hợp bằng **Reciprocal Rank Fusion** — điểm cuối của mỗi tài liệu là `Σ 1 / (k + rank_i)`. Cách này cân bằng giữa độ nhớ theo từ khoá chính xác (sparse) và độ nhớ ngữ nghĩa (dense).

### 4.2 Reranking

Top-k tài liệu truy xuất được (mặc định 20) được chấm điểm lại bằng `BAAI/bge-reranker-v2-m3` (cross-encoder, từng cặp query+passage). Điểm được chuẩn hoá bằng sigmoid. Chỉ giữ top-n (mặc định 5).

### 4.3 Policy Web Fallback

`should_add_web_fallback()` trong `rag_policy.py` kích hoạt tìm kiếm web khi corpus nội bộ không đủ mạnh:

| Điều kiện | Nhãn lý do |
|-----------|-----------|
| Không có tài liệu nào sau rerank | `no_docs` |
| Điểm cao nhất < 0.70 | `hard_below` |
| Điểm cao nhất < 0.85 **và** hỗ trợ mỏng (< 2 docs, khoảng cách ≥ 0.12) | `shallow` |
| Phát hiện dấu hiệu thời sự ("hôm nay", "mới nhất", "current", "latest") | `time_sensitive` |
| Câu hỏi mở rộng đáng kể (thêm 4+ token so với gốc) | `expansion` |
| Điểm cao nhất ≥ 0.85 và đủ hỗ trợ | bỏ qua — `soft_above` |
| Không có Tavily key | bỏ qua — `disabled` |

### 4.4 Hệ thống trích dẫn

```text
LLM tạo ra:     "GDP tăng 6.5% [S1] nhờ xuất khẩu mạnh [S2]."
                           ↓ finalize_citations()
Câu trả lời:    "GDP tăng 6.5% [[S1]](https://...) nhờ xuất khẩu mạnh [[S2]](https://...)."
                           +
                 ### Nguồn trích dẫn
                 - [S1] Tiêu đề nguồn 1 — điểm: 0.92
                 - [S2] Tiêu đề nguồn 2 — điểm: 0.87
```

`rewrite_inline_citations()` dùng regex và bỏ qua các khối code có hàng rào để tránh làm hỏng ví dụ code.

---

## 5. Cấu trúc thư mục

```text
.
├── Makefile                         # Mọi lệnh dev & ops
├── pyproject.toml                   # Phụ thuộc Python (uv)
├── .env.example                     # Tất cả biến cấu hình với giá trị mặc định
├── docker-compose.yaml              # Stack production
├── docker-compose.dev.yaml          # Stack dev (bind mounts, hot reload)
│
├── api/
│   └── orchestrator/
│       ├── config.py                # Pydantic Settings (lồng nhau, env-driven)
│       ├── main.py                  # FastAPI app factory + lifespan clients
│       ├── tracing.py               # Loguru + LangSmith setup
│       ├── models/
│       │   └── schemas.py           # Schema message tương thích OpenAI
│       ├── pipeline/
│       │   ├── rag.py               # Định nghĩa LangGraph StateGraph
│       │   ├── rag_policy.py        # Heuristic web fallback
│       │   ├── rag_context.py       # Gộp context + hậu xử lý trích dẫn
│       │   └── rag_prompts.py       # Template prompt (tiếng Việt)
│       ├── routers/
│       │   └── chat.py              # POST /v1/chat/completions
│       └── services/
│           ├── llm.py               # LLMClient (OpenAI async wrapper)
│           ├── inference.py         # InferenceClient (embed/sparse/rerank)
│           ├── retriever.py         # RetrieverClient (Qdrant hybrid search)
│           ├── web_search.py        # WebSearchClient (Tavily)
│           └── conversation.py      # Chuẩn hoá message
│
├── services/
│   └── inference/
│       ├── inference_app.py         # FastAPI: /embed /sparse /rerank /health
│       └── requirements.txt
│
├── scripts/
│   ├── ingest.py                    # Dataset HuggingFace → Qdrant
│   ├── chunker.py                   # Phân đoạn ngữ nghĩa
│   ├── qdrant_bootstrap.py          # Tạo collection + vector indexes
│   └── qdrant_snapshot_restore.py   # Khôi phục từ file .snapshot
│
├── infra/docker/
│   ├── app.cpu.Dockerfile           # Orchestrator + ingest (CPU)
│   └── app.gpu.Dockerfile           # Inference service (GPU, CUDA 12.8)
│
├── data/
│   └── *.snapshot                   # Snapshot Qdrant để tái tạo dữ liệu
│
└── tests/
    └── orchestrator/                # Unit + integration tests
```

---

## 6. Các dịch vụ runtime

| Service | Port | GPU | Mô tả |
|---------|------|-----|-------|
| `orchestrator` | 8000 | — | FastAPI + pipeline LangGraph |
| `inference` | 8001 | GPU 0 | Embedding + sparse encoding + reranking |
| `llm` |  | GPU 1 | vLLM endpoint tương thích OpenAI (thuê qua Vast.ai) |
| `qdrant` | 6333 / 6334 | — | Cơ sở dữ liệu vector (REST / gRPC) |
| `webui` | 8080 | — | Giao diện chat Open WebUI |
| `bootstrap` | — | — | Một lần: tạo Qdrant collection (profile: `tools`) |
| `ingest` | — | — | Một lần: nạp dữ liệu vào Qdrant (profile: `ingest`) |
| `tunnel` | — | — | Cloudflare tunnel (tuỳ chọn) |

Service `llm` bị comment mặc định — chỉ cần trỏ `LLM__URL` đến instance vLLM ngoài.

---

## 7. Khái niệm & Lựa chọn thiết kế

### 7.1 Định tuyến ý định (Direct vs. RAG)

**Khái niệm:** Không phải câu hỏi nào cũng cần RAG. Một số là kiến thức tổng quát ("GDP là gì?") hoặc thoại chuyện ("Chào bạn!"). Hệ thống dùng LLM để phân loại mỗi câu hỏi:

- **Direct:** Trả lời ngay từ kiến thức LLM
- **Direct + Web:** Trả lời direct + bổ sung kết quả web trực tiếp
- **RAG:** Pipeline truy xuất đầy đủ cho câu hỏi miền đặc thù

**Lựa chọn thiết kế:** Này tiết kiệm latency retrieval+reranking cho ~30% câu hỏi, đồng thời cải thiện độ tươi mới của câu trả lời cho câu hỏi thời sự. Bộ phân loại chạy inline — không cần service riêng.

### 7.2 Truy xuất Hybrid (Dense + Sparse + RRF)

**Khái niệm:** Hai chế độ tìm kiếm bổ sung:

- **Dense embeddings** (BAAI/bge-m3): Bắt được ý nghĩa ngữ nghĩa. "GDP tăng trưởng" khớp với "mở rộng kinh tế" qua 1024-dim similarity.
- **Sparse lexical** (BM25 weights): Khớp từ khoá chính xác. "GDP tăng trưởng" khớp với "GDP tăng trưởng" từng chữ.

Cái nào một mình cũng không đủ. Câu hỏi về "Thị trường chứng khoán Việt Nam năm 2026" cần cả khớp ngữ nghĩa ("tâm lý thị trường") và từ khoá chính xác ("2026").

**Reciprocal Rank Fusion (RRF):** Kết hợp cả hai ranking qua `Σ 1 / (k + rank_i)` — không cần tham số, ổn định. Tài liệu ranked #1 ở dense và #10 ở sparse sẽ có điểm kết hợp mạnh.

**Lựa chọn thiết kế:** Hybrid search tránh phải tinh chỉnh tham số (không alpha blend factor) và cung cấp bảo hiểm — nếu mode nào đó thất bại, mode kia bù đắp phần nào.

### 7.3 Policy Web Fallback (Bổ sung thích ứng)

**Khái niệm:** Corpus nội bộ thưa (~50k bài báo kinh tế). Khi độ tin cậy retrieval thấp, tìm kiếm web bổ sung:

| Tín hiệu | Ý nghĩa |
| --- | --- |
| `hard_below` (score < 0.70) | Khớp yếu → web search bù đắp |
| `shallow` (score < 0.85 + nguồn mỏng) | Khớp sơ sài với chỉ 1–2 nguồn → web cung cấp đa dạng |
| `time_sensitive` ("hôm nay", "2026") | Độ tươi mới quan trọng → lấy kết quả web mới |
| `expansion` (thêm 4+ token) | Câu hỏi thay đổi đáng kể → corpus có thể lỗi thời |

**Lựa chọn thiết kế:** Fallback là policy-driven, không threshold-driven. Thay vì hard cutoff ở 0.75, ta hỏi "điều này đủ tốt không?" — xem xét không chỉ score mà context depth, tươi mới, và expansion. Điều này giảm false positives (spam web) trong khi bắt được khoảng trống thực.

### 7.4 Reranking với Cross-Encoders

**Khái niệm:** Dense retrieval trả top-20 ứng viên nhanh. Nhưng ranking bằng similarity thô — tất cả 20 docs có thể "tương tự". Reranking dùng cross-encoder (`BAAI/bge-reranker-v2-m3`), mã hoá query+passage chung (không tách riêng như dense). Điều này tinh chỉnh top-5:

```text
Trước rerank: [News1:0.82, News2:0.81, News3:0.79, News4:0.78, News5:0.77, Spam:0.76]
Sau rerank:   [News1:0.92, News3:0.87, News2:0.71, Spam:0.45]  ← ranking chính xác
```

**Lựa chọn thiết kế:** Cross-encoders chậm hơn (vài ms/cặp) nhưng xứng đáng cho ~8 docs gửi đến LLM. Dense-only ranking có thể lật ngược kết quả top bất ngờ; reranking ổn định final context.

### 7.5 Hệ thống trích dẫn với hậu xử lý

**Khái niệm:** LLM tạo câu trả lời với marker nội tuyến `[S1]`, `[S2]`. Những marker này heuristic (không đảm bảo khớp retrieved sources). Hậu xử lý:

1. Trích xuất `[Sn]` markers → kiểm tra citation_pool lấy metadata
2. Viết lại thành markdown có thể click `[[S1]](url)`
3. Thêm section "Nguồn trích dẫn" với tiêu đề, URL, và điểm tin cậy

**Lựa chọn thiết kế:** Inline citations minh bạch hơn footnotes. Chúng chỉ ra câu nào dựa trên nguồn nào. Hậu xử lý đảm bảo tất cả trích dẫn hợp lệ (checks 404 bỏ qua vì tốc độ, nhưng URL xác minh lúc ingest).

### 7.6 LangGraph thay vì Python trực tiếp

**Khái niệm:** LangGraph là state machine framework. Thay vì:

```python
# ❌ Hàm monolithic
def rag_pipeline(query):
    embed = embed_query(query)
    docs = retrieve(embed)
    ...
```

Dùng:

```python
# ✅ Nodes có thể kết hợp
graph = StateGraph(RAGState)
graph.add_node("embed", embed_node)
...
```

**Lựa chọn thiết kế:** Cái này cho phép:

- **Kiểm tra:** Xem đồ thị trực quan trong LangSmith
- **Test:** Mock từng node riêng
- **Streaming:** Tách retrieval khỏi generation (chế độ retrieval_only)
- **Observability:** Tự động trace từng node

### 7.7 Dedicated Inference Service

**Khái niệm:** Embedding + reranking là GPU-bound. Orchestrator là CPU-bound. Tách chúng:

```text
Orchestrator (CPU)  ←→  Inference (GPU 0)  +  LLM (GPU 1)
      ↑                         ↑
  Route request            Embedding, sparse, rerank
  Tương tác flow           (tái sử dụng cho replicas)
```

**Lựa chọn thiết kế:** Một inference service phục vụ 10 replica orchestrator. Cách này khấu hao chi phí GPU và cho phép scaling độc lập. Nếu embedding chậm, thêm GPU vào inference mà không sửa orchestrator.

### 7.8 AWQ Quantization cho Qwen

**Khái niệm:** Qwen-4B trong FP16 là ~8 GB. AWQ quantization giảm xuống ~3 GB với mất chất lượng tối thiểu (perplexity drop < 0.5%). Vừa vặn trên một GPU consumer.

**Lựa chọn thiết kế:** Quantization đánh đổi latency (chậm 2–5%) để lấy VRAM. Vì model 4B đã nhanh (< 100ms/token), cách đánh đổi này xứng đáng cho chi phí và sẵn có.

### 7.9 Tương thích OpenAI

**Khái niệm:** Endpoint `/v1/chat/completions` tuân theo spec OpenAI:

```json
{
  "model": "Economic-RAG-Assistant",
  "messages": [{"role": "user", "content": "Tỷ giá hiện tại là bao nhiêu?"}],
  "stream": false
}
```

Bất kỳ client nào dùng `openai-python`, `curl`, hoặc Open WebUI đều kết nối được không cần sửa.

**Lựa chọn thiết kế:** Tương thích giảm lock-in và cho phép dễ dàng swap với LLM khác. Client không bị ràng buộc vào dự án này.

---

## 8. Tham khảo cấu hình

Tất cả biến dùng dấu phân cách lồng nhau `__` (ví dụ: `LLM__MODEL` ánh xạ đến `settings.llm.model`).

```env
# ── LLM ──────────────────────────────────────────────────────────────────
LLM__URL=http://llm:8004/v1
LLM__MODEL=quannguyen204/Qwen3-4B-Instruct-2507-AWQ-W4A16
LLM__TEMPERATURE=0.7
LLM__MAX_TOKENS=1024
LLM__TIMEOUT=60.0
LLM__API_KEY=dummy

# ── vLLM runtime ──────────────────────────────────────────────────────────
VLLM__MAX_MODEL_LEN=16384
VLLM__GPU_MEMORY_UTILIZATION=0.95
VLLM__QUANTIZATION=compressed-tensors

# ── Services ──────────────────────────────────────────────────────────────
SERVICES__INFERENCE_URL=http://inference:8001
SERVICES__QDRANT_URL=http://qdrant:6333
SERVICES__QDRANT_COLLECTION=academic_chunks

# ── Tinh chỉnh RAG ────────────────────────────────────────────────────────
RAG__RETRIEVAL_TOP_K=20
RAG__RERANK_TOP_N=5
RAG__WEB_FALLBACK_HARD_THRESHOLD=0.70
RAG__WEB_FALLBACK_SOFT_THRESHOLD=0.85
RAG__CONTEXT_LIMIT=5
RAG__CITATION_LIMIT=5

# ── Ingest ────────────────────────────────────────────────────────────────
INGEST__FORCE_RECREATE=false
SNAPSHOT_HF_REPO=quannguyen204/economic-rag-snapshots
SNAPSHOT_FILENAME=academic_chunks-*.snapshot

# ── Observability ─────────────────────────────────────────────────────────
OBSERVABILITY__LOG_LEVEL=INFO
OBSERVABILITY__LANGSMITH_API_KEY=
OBSERVABILITY__TAVILY_API_KEY=       # Web search fallback (tuỳ chọn)

# ── Auth & infra ──────────────────────────────────────────────────────────
HF_TOKEN=                            # Bắt buộc để tải model
DOCKERHUB_NAMESPACE=your-namespace
WEBUI_SECRET_KEY=admin
CLOUDFLARE_TUNNEL_TOKEN=             # Tuỳ chọn
```

---

## 9. Cài đặt & Khởi động nhanh

### Yêu cầu hệ thống

- Docker với plugin `docker compose`
- NVIDIA Container Toolkit (cho GPU services)
- `uv` — quản lý package Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Phương án A — Full stack từ snapshot (khuyến nghị)

```bash
# 1. Clone và cấu hình
git clone https://github.com/minhquana1906/Multimodal-Economic-RAG-Assistant.git
cd Multimodal-Economic-RAG-Assistant
make setup                  # copy .env.example → .env
# Sửa .env: đặt HF_TOKEN và tuỳ chọn OBSERVABILITY__TAVILY_API_KEY

# 2. Khởi động tất cả (pull image, khôi phục snapshot, start services)
make start

# 3. Kiểm tra
curl http://localhost:8000/health
curl http://localhost:6333/collections
# Mở WebUI tại http://localhost:8080
```

### Phương án B — Dev stack (hot reload)

```bash
make setup                  # copy .env.example → .env
# Sửa .env
make dev                    # build image từ source với bind mounts
# Sửa file trong api/ hoặc services/inference/ — restart container để áp dụng
make dev-logs               # xem tất cả logs
make dev-stop
```

### Các lệnh Make phổ biến

| Lệnh | Mô tả |
|------|-------|
| `make start` | Full stack: pull → khôi phục snapshot → up |
| `make stop` | Dừng tất cả containers |
| `make dev` | Dev stack với bind mounts |
| `make dev-stop` | Dừng dev stack |
| `make logs [SERVICE]` | Theo dõi logs |
| `make ps` | Xem trạng thái containers |
| `make test` | Chạy unit tests |
| `make test-integration` | Chạy integration tests |
| `make bootstrap` | Tạo Qdrant collection cục bộ |
| `make snapshot-restore` | Khôi phục Qdrant từ .snapshot |
| `make build` | Build Docker images cục bộ |
| `make push` | Đẩy images lên Docker Hub |

---

## 10. Quy trình phát triển

```bash
# Môi trường Python cục bộ
uv sync --dev

# Chạy tests
uv run pytest                         # unit tests
uv run pytest -m integration          # integration tests

# Dev compose (hot reload qua bind mounts)
make dev
make dev-logs orchestrator            # xem log orchestrator
make restart orchestrator             # áp dụng thay đổi code
```

Logs dùng các cấp độ theo domain (`RETRIEVAL`, `RERANK`, `LLM`) để dễ lọc.

LangSmith tracing tuỳ chọn — đặt `OBSERVABILITY__LANGSMITH_API_KEY` để bật visualize trace đầu-cuối.

---

## 11. Kiểm thử

Test suite bao gồm:

| File test | Phạm vi |
|-----------|---------|
| `test_rag_pipeline.py` | Các node đồ thị, chuyển đổi trạng thái |
| `test_rag_helpers.py` | Quyết định policy, viết lại trích dẫn |
| `test_chat_flow.py` | Định tuyến chat đầu-cuối |
| `test_chat_router.py` | Hành vi HTTP endpoint |
| `test_llm.py` | Các phương thức LLMClient |
| `test_retriever.py` | Hybrid search, fallback về dense |
| `test_inline_citations.py` | Hậu xử lý trích dẫn |
| `test_regression_matrix.py` | Câu hỏi thực tế, follow-up, từ khoá thưa, no-context, web, streaming |

```bash
uv run pytest                         # tất cả unit tests
uv run pytest tests/orchestrator/test_regression_matrix.py -m integration
```

---

## 12. Xử lý sự cố

### Services không bao giờ healthy

- Kiểm tra `HF_TOKEN` đã đặt và hợp lệ (bắt buộc để tải model).
- Xác minh NVIDIA Container Toolkit cài đặt: `nvidia-smi` trong container.
- Kiểm tra VRAM GPU — inference cần ~4 GB, LLM cần ~6–8 GB.

### Câu trả lời thất bại / lỗi LLM

- Xác minh `LLM__URL` và `LLM__MODEL` khớp với instance vLLM đang chạy.
- Kiểm tra `make logs llm` để xem lỗi khởi động vLLM.

### Không có kết quả truy xuất

- Xác nhận Qdrant collection tồn tại: `curl http://localhost:6333/collections`.
- Chạy `make bootstrap` rồi `make ingest` (hoặc `make snapshot-restore`).

### Ingest thất bại / xung đột collection

- Đặt `INGEST__FORCE_RECREATE=true` để xoá và tạo lại collection (phá huỷ dữ liệu).

### Web search không kích hoạt

- Đặt `OBSERVABILITY__TAVILY_API_KEY` trong `.env`. Nếu không có key, web fallback bị tắt âm thầm.

---

> 🇻🇳 **Tiếng Việt** | [🇬🇧 English](README.md)
