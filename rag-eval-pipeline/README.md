# HealthAgent MCP

An agentic AI system for healthcare organisations, serving two distinct roles from a single platform: **HR staff** and **patients**. Built with FastMCP, LangChain, and Langfuse.

## What it does

When a user logs in, the system determines their role and initialises an LangChain agent loaded with only the tools relevant to that role. Every LLM call is traced in Langfuse with faithfulness and relevance scores via LLM-as-a-judge evaluation.

```
User logs in → role detected (HR or Patient)
    → LangChain agent initialised with role-specific MCP tools
    → FastAPI handles requests
    → Langfuse traces every LLM call + runs eval scoring
```

### HR agent tools
| Tool | Description |
|---|---|
| `get_employee_profile` | Fetch staff record by employee ID |
| `check_compliance_status` | Check certification and licence expiry |
| `get_workforce_insights` | Headcount and open roles by department |
| `query_hr_policy` | RAG over HR policy knowledge base |

### Patient agent tools
| Tool | Description |
|---|---|
| `get_upcoming_appointments` | List scheduled appointments |
| `get_medical_summary` | Conditions, allergies, blood type |
| `get_medication_reminders` | Active medications with dosage instructions |
| `query_care_plan` | RAG over care plan knowledge base |

## Stack

- **Agent orchestration:** LangChain (OpenAI tools agent)
- **MCP server:** FastMCP
- **LLM evaluation & tracing:** Langfuse (LLM-as-a-judge, faithfulness, relevance)
- **Vector store:** ChromaDB (local) or Milvus/Zilliz (production)
- **Embeddings:** OpenAI text-embedding-3-small
- **API:** FastAPI + JWT auth
- **LLM:** GPT-4o (configurable)

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/Aurovindhya/healthagent-mcp.git
cd healthagent-mcp
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in your API keys in .env
```

Required keys:
- `OPENAI_API_KEY` — for LLM and embeddings
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` — from [cloud.langfuse.com](https://cloud.langfuse.com)

### 3. Ingest sample documents

```bash
python ingest.py
```

This loads HR policy and care plan documents into the vector store for RAG.

### 4. Run the API

```bash
uvicorn src.api.app:app --reload
```

API docs available at `http://localhost:8000/docs`

## Example usage

### Login as HR

```bash
curl -X POST http://localhost:8000/auth/login \
  -d "username=hr_user&password=hrpass123"
```

### Chat as HR

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Check compliance status for EMP001", "session_id": "<session_id>"}'
```

### Login as Patient

```bash
curl -X POST http://localhost:8000/auth/login \
  -d "username=patient_user&password=patientpass123"
```

### Chat as Patient

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are my upcoming appointments?", "session_id": "<session_id>"}'
```

## Langfuse evaluation

Every response is automatically scored by an LLM-as-a-judge evaluator:

- **Faithfulness** (RAG queries): how grounded the answer is in retrieved context
- **Relevance**: how directly the answer addresses the question

Scores are pushed to your Langfuse dashboard in real time. You can view traces, compare prompt versions, and track score regressions over time.

## Switching to Milvus / Zilliz

Set `VECTOR_STORE=milvus` in `.env` and provide your `ZILLIZ_URI` and `ZILLIZ_TOKEN`. No other changes needed — the retrieval layer handles both backends.

## Project structure

```
healthagent-mcp/
├── src/
│   ├── agent.py              # LangChain agent, role-based tool loading
│   ├── auth.py               # JWT auth, user roles
│   ├── config.py             # Settings from .env
│   ├── api/
│   │   └── app.py            # FastAPI endpoints
│   ├── mcp_tools/
│   │   ├── hr_tools.py       # HR MCP tools
│   │   └── patient_tools.py  # Patient MCP tools
│   ├── retrieval/
│   │   └── vector_store.py   # Chroma / Milvus abstraction
│   └── evaluation/
│       └── langfuse_eval.py  # LLM-as-a-judge scoring
├── data/
│   └── sample_docs/          # HR policies and care plans
├── ingest.py                 # One-time document ingestion script
├── requirements.txt
└── .env.example
```
