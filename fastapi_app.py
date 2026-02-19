"""
fastapi_app.py: Serve the RAG system using FastAPI for API access.
"""
from rag import demo, llm_with_chat_history, current_vector_store, get_eval
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import gradio as gr
from pydantic import BaseModel

app = FastAPI(title="Mini RAG FastAPI Endpoint")

DEFAULT_SCORE_THRESHOLD = 0.2

class QueryRequest(BaseModel):
    query: str
    score_threshold: float | None = None


@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

@app.post("/rag")
async def rag_endpoint(request: QueryRequest):
    if current_vector_store is None:
        raise HTTPException(status_code=400, detail="No PDF has been loaded. Upload a PDF in the Gradio app first.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Please provide a non-empty query.")

    score_threshold = request.score_threshold if request.score_threshold is not None else DEFAULT_SCORE_THRESHOLD

    results = current_vector_store.similarity_search_with_relevance_scores(
        query,
        k=5,
        score_threshold=score_threshold,
    )
    if not results:
        context_text = "No relevant context found."
    else:
        context_text = "\n\n---\n\n".join([
            f"Page: {doc[0].metadata.get('page', 'Unknown')}\nContent:\n{doc[0].page_content}"
            for doc in results
        ])
    response = llm_with_chat_history.invoke(
        {"question": query, "context_text": context_text},
        config={"configurable": {"session_id": "user1"}}
    )
    eval_result = get_eval(query, context_text, response)
    return {"response": response, "evaluation": eval_result}

# Mount the Gradio UI under /ui to keep /rag working.
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
