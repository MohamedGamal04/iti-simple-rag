"""
langserve_app.py: Serve the RAG chain using LangServe for API access.
"""
from rag import llm_with_chat_history, current_vector_store, get_eval
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="Mini RAG LangServe API")

# Example endpoint for RAG chat
@app.post("/rag")
async def rag_endpoint(query: str):
    results = vector_store.similarity_search_with_relevance_scores(query, k=5, score_threshold=0.2)
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

# Add LangServe routes for the chain (optional, for more advanced use)
add_routes(app, llm_with_chat_history, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
