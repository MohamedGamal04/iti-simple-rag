"""
fastapi_app.py: Serve the RAG system using FastAPI for API access.
"""
from rag import llm_with_chat_history, current_vector_store, get_eval
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mini RAG FastAPI Endpoint")

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
async def rag_endpoint(request: QueryRequest):
    query = request.query
    results = current_vector_store.similarity_search_with_relevance_scores(query, k=5, score_threshold=0.2)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
