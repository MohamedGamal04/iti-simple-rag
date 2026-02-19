# ITI RAG

Simple Retrieval-Augmented Generation (RAG) app with:
- Gradio UI for PDF upload + chat
- FastAPI endpoint (`/rag`)
- Optional LangServe routes (`/chain`)
- Chroma persistent vector store

## Requirements

- Python 3.10+
- NVIDIA API key

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_api_key_here
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### 1) Gradio-only app
```bash
python rag.py
```

### 2) FastAPI + Gradio mounted at `/ui`
```bash
python fastapi_app.py
```
- API endpoint: `POST /rag`
- UI: `/ui`

### 3) LangServe + Gradio mounted at `/`
```bash
python langserve_app.py
```
- API endpoint: `POST /rag`
- LangServe chain routes: `/chain`

## Example API Request

```bash
curl -X POST http://127.0.0.1:8001/rag \
	-H "Content-Type: application/json" \
	-d '{"query":"Summarize the document","score_threshold":0.2}'
```

## Notes

- Upload a PDF first from the UI before querying `/rag`.
- Chroma data is stored in `./chroma_langchain_db`.
