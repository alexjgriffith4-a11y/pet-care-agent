# Pet Care Support Agent

A retrieval-augmented generation (RAG) agent that answers dog and cat care questions grounded in trusted veterinary sources. Built as a course project for **RSM 8430**

## What It Does

The agent accepts natural-language questions about pet nutrition, health symptoms, food/toxin safety, and daily care routines, then returns grounded answers with source citations. It is **not** a veterinarian — every health-related response includes a medical disclaimer directing users to consult a licensed vet.

### Core Capabilities


| Capability                 | Description                                                                                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Food & Toxin Safety**    | Single-turn check: "Can my dog eat grapes?" → risk level (SAFE / CAUTION / TOXIC / UNKNOWN) with cited sources and poison-control contact info.                                         |
| **Symptom Triage**         | Multi-turn flow that collects species, symptoms, duration, and weight across turns, then returns an urgency classification (MONITOR / VET SOON / VET NOW) with a veterinary disclaimer. |
| **General Pet Care Q&A**   | RAG-based answers for any in-scope dog/cat question — grooming, vaccinations, behaviour, senior care, etc.                                                                              |
| **Out-of-Scope Rejection** | Politely declines questions unrelated to pet care (weather, coding, poetry, etc.).                                                                                                      |
| **Guardrails**             | Prompt injection detection, harmful-request blocking, retrieval confidence checks, output safety filters, and mandatory medical disclaimers.                                            |


## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Input Guardrails  │  prompt injection, harmful content, domain check
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Intent Classifier  │  few-shot Qwen3 → food_safety | symptom_triage |
│  (intent_classifier)│                    care_routine | general_qa |
└─────────┬───────────┘                    out_of_scope
          ▼
┌─────────────────────┐
│    Action Router    │  agent.py routes to the appropriate handler
│      (agent.py)     │
└──┬──────┬──────┬────┘
   │      │      │
   ▼      ▼      ▼
┌──────┐┌──────┐┌──────────┐
│Food  ││Sympt.││General QA│   Each handler retrieves chunks from
│Safety││Triage││  (RAG)   │   ChromaDB, builds a prompt, and calls
└──┬───┘└──┬───┘└──┬───────┘   the Qwen3 LLM.
   │       │       │
   ▼       ▼       ▼
┌─────────────────────┐
│   Retriever + RAG   │  ChromaDB vector store + bge-base-en-v1.5 embeddings
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Output Guardrails  │  citation check, medical disclaimer, prompt leak
└─────────┬───────────┘  detection, unsafe content filter
          ▼
     Agent Response
   (text + sources + intent)
```

## Project Structure

```
pet-care-agent/
├── agent.py                 # Agent loop — single entry point for every user turn
├── intent_classifier.py     # Few-shot Qwen3 intent classification (5 labels)
├── retriever.py             # ChromaDB query interface (retrieve + format context)
├── embedder_api.py          # Client for the bge-base-en-v1.5 embeddings endpoint
├── guardrails.py            # Input, retrieval, and output safety policies
│
├── actions/
│   ├── food_safety.py       # Single-turn food/toxin risk assessment
│   ├── symptom_triage.py    # Multi-turn symptom collection + urgency classification
│   └── profiles.json        # Pet profile storage (name, species, breed, age, weight)
│
├── data/
│   ├── raw/                 # 355 scraped articles (JSON) from PetMD and ASPCA
│   ├── chunks/
│   │   └── all_chunks.json  # ~1,597 text chunks with metadata
│   └── vectorstore/         # ChromaDB persistent store (auto-generated)
│
├── evaluation/
│   ├── test_cases.py        # 15 test cases across 6 categories
│   ├── results.md           # Auto-generated evaluation report (15/15 passing)
│   └── __init__.py
│
├── aspca_scraper.ipynb      # Web scraper for ASPCA pet care articles
├── petmd_scraper.ipynb      # Web scraper for PetMD pet health articles
├── chunker.ipynb            # Splits raw articles into ~400-token chunks
├── embedder.ipynb           # Embeds chunks and populates ChromaDB
├── retriever.ipynb          # Interactive notebook for testing retrieval
│
├── logs/
│   └── guardrail_events.jsonl  # Structured guardrail decision log
│
├── requirements.txt
├── project_plan.md          # Internal project plan and phase breakdown
└── .gitignore
```

## Data Pipeline

The data pipeline runs once to populate the vector store. It is implemented as a series of Jupyter notebooks:

1. **Scraping** (`aspca_scraper.ipynb`, `petmd_scraper.ipynb`) — Collects pet care articles from ASPCA and PetMD. Respects `robots.txt` with polite request delays. Outputs structured JSON files to `data/raw/` with the schema `{title, content, url, source, species, topic}`.
2. **Chunking** (`chunker.ipynb`) — Splits each article into ~~400-token chunks (~~1,600 characters) with 200-character overlap using LangChain's `RecursiveCharacterTextSplitter`. Strips boilerplate text. Preserves per-chunk metadata. Outputs `data/chunks/all_chunks.json` (~1,597 chunks).
3. **Embedding** (`embedder.ipynb`) — Embeds all chunks using the **bge-base-en-v1.5** model via the A2 course endpoint (768-dimensional vectors). Stores embeddings + metadata in a local ChromaDB persistent collection (`data/vectorstore/`).

## Tech Stack


| Component    | Technology                                            |
| ------------ | ----------------------------------------------------- |
| LLM          | Qwen3-30B-A3B (FP8), hosted course endpoint           |
| Embeddings   | bge-base-en-v1.5 via A2 course endpoint (768-dim)     |
| Vector Store | ChromaDB (local persistent storage)                   |
| Framework    | LangChain (ChatOpenAI, message types, text splitters) |
| Scraping     | requests + BeautifulSoup4                             |
| UI           | Streamlit (planned)                                   |
| Language     | Python 3.12                                           |


## Setup

### Prerequisites

- Python 3.10+
- Access to the RSM 8430 course API endpoints (Qwen3 + A2 embedder)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd pet-care-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
RSM_API_KEY=<your-course-api-key>
```

### Populating the Vector Store

If the `data/vectorstore/` directory is not present (it is excluded from git), run the data pipeline notebooks in order:

1. `petmd_scraper.ipynb` — scrape PetMD articles
2. `aspca_scraper.ipynb` — scrape ASPCA articles
3. `chunker.ipynb` — chunk all raw articles
4. `embedder.ipynb` — embed chunks into ChromaDB

### Running the Agent

The agent is invoked programmatically via `agent.run_turn()`:

```python
from agent import run_turn

result = run_turn("Can my dog eat chocolate?")
print(result["response"])   # Grounded answer with risk level
print(result["sources"])    # [{title, url, score}, ...]
print(result["intent"])     # "food_safety"
```

### Running the Evaluation Suite

```bash
python -m evaluation.test_cases
```

This runs 15 test cases across six categories (retrieval, routing, actions, out-of-scope rejection, prompt injection, error handling) and writes a detailed report to `evaluation/results.md`.

## Evaluation Results

The most recent evaluation run achieved **15/15 passing** (0 failures, 0 errors, 0 skips):


| Category         | Pass | Fail | Total |
| ---------------- | ---- | ---- | ----- |
| Retrieval        | 3    | 0    | 3     |
| Routing          | 3    | 0    | 3     |
| Actions          | 3    | 0    | 3     |
| Out-of-Scope     | 2    | 0    | 2     |
| Prompt Injection | 2    | 0    | 2     |
| Error Handling   | 2    | 0    | 2     |


See `evaluation/results.md` for per-case details and the failure analysis section.

## Guardrails

The agent implements layered safety controls in `guardrails.py`:

- **Input guardrails** — Prompt injection detection (regex + unicode normalization), harmful intent blocking, domain scope enforcement, query length limits, and emergency-context-aware overrides (e.g., "my dog ate poison" is allowed through).
- **Retrieval guardrails** — Minimum chunk count, top-score threshold, and average-score threshold prevent the agent from generating answers when retrieved evidence is weak.
- **Output guardrails** — Blocks prompt leakage, unsafe operational instructions, and responses missing required source citations or medical disclaimers. Includes an auto-fix layer that appends disclaimers when possible before falling back to a hard block.
- **Logging** — All guardrail decisions are logged to `logs/guardrail_events.jsonl` with timestamps and hashed query identifiers (no raw query text stored).

## Known Limitations

- **Pet profile / care routine** — `actions/pet_profile.py` is not yet implemented. The `care_routine` intent falls back to a stub message. This is documented as a Phase 3 gap.
- **Streamlit UI** — The `app.py` chat interface is planned but not yet built.
- **Single-endpoint dependency** — Both the LLM (Qwen3) and the embedder (bge-base-en-v1.5) rely on course-hosted endpoints. The agent is not functional without them.
- **No persistent conversation state** — Session history must be passed by the caller on every turn. There is no built-in session store.
- **Heuristic species detection** — Species is detected via keyword matching, not NER. Queries without explicit species keywords search across both dogs and cats.

## License

Course project — not licensed for production use.