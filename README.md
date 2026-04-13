# Le Sommelier — Voice Wine Explorer

A voice-enabled wine Q&A app backed by RAG (Retrieval-Augmented Generation).

## Stack
- **Backend**: FastAPI · LangChain LCEL · Chroma · HuggingFace `all-mpnet-base-v2` · Claude `claude-haiku-4-5-20251001`
- **Frontend**: Single HTML page · Web Speech API (STT + TTS)

---

## Setup

### 1. Place your wine data
Copy your `wines.csv` file into `backend/`:

```
voice-wine-explorer/backend/wines.csv
```

Expected columns: any flat fields (name, varietal, region, vintage, price, …) plus an optional `professional_ratings` column containing a JSON array like:

```json
[{"source": "Wine Spectator", "score": 94, "note": "Rich and layered..."}]
```

### 2. Configure environment

```bash
cd backend
cp .env.example .env
# Fill in your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   WINE_CSV_PATH=./wines.csv
```

### 3. Install Python dependencies

```bash
cd backend
.venv\Scripts\Activate.ps1       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


### 4. Start the backend
 
```bash
uvicorn main:app --reload --port 8000
```
 
On first run, Chroma will embed all wine rows and persist them to `./chroma_db/`.
**This can take a few minutes on the first startup.** Wait for the message:
 
```
INFO:     Application startup complete.
```
 
before sending any requests. Subsequent starts skip embedding and load directly from disk.

### 5. Open the app

FastAPI serves both the frontend and API. Open:

```text
http://localhost:8000/
```

In GitHub Codespaces, open the forwarded URL for port `8000`.

---

## Usage

| Action | Result |
|--------|--------|
| Tap the orb | Start voice recognition |
| Tap again | Stop and send question |
| Click a pill | Submit example question |
| Answer plays automatically | Via browser TTS |

---

## Project Structure

```
voice-wine-explorer/
├── backend/
│   ├── main.py           # FastAPI app + /ask endpoint
│   ├── rag.py            # Chroma build/load + LCEL chain
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html        # Single-page voice UI
└── README.md
```
