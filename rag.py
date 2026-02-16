import os
import dotenv
import warnings
import gradio as gr
import chromadb
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
dotenv.load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    print("Please check or regenerate your API key.")
    exit(1)

# Initialize Models
embedder = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    truncate="NONE",
)

llm = ChatNVIDIA(
    model="meta/llama-3.1-405b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
    streaming=False,
)

# Initialize Chroma Client
client = chromadb.PersistentClient(path="./chroma_langchain_db")
current_vector_store = None

# --- Helper Functions ---

def process_uploaded_pdf(file_path):
    global current_vector_store
    if not file_path:
        return "No file uploaded."

    try:
        # Load PDF
        loader = PyMuPDFLoader(file_path)
        full_doc = loader.load()
        
        # Clean
        docs = [page for page in full_doc if page.page_content.strip()]
        if not docs:
            return "Error: PDF contains no readable text."

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = []

        for doc in docs:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)

        original_filename = os.path.basename(file_path)
        pdf_name = os.path.splitext(original_filename)[0]
        col_name = f"col_{pdf_name}".replace(" ", "_").replace("-", "_").lower()
        
        existing_collections = [col.name for col in client.list_collections()]
        
        if col_name in existing_collections:
            current_vector_store = Chroma(
                collection_name=col_name,
                embedding_function=embedder,
                persist_directory="./chroma_langchain_db",
            )
            msg = f"Loaded existing collection '{col_name}' ({current_vector_store._collection.count()} docs)."
        else:
            current_vector_store = Chroma.from_documents(
                documents=chunks,
                collection_name=col_name,
                embedding=embedder,
                persist_directory="./chroma_langchain_db",
            )
            msg = f"Created new collection '{col_name}' with {len(chunks)} chunks."
            
        return msg

    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def get_eval(query, context_text, response):
    eval_prompt = f"""
    INSTRUCTION: Evaluate the RAG response (0=Refusal, 1=Hallucination, 2=Success).
    You are given:
    - The user's current question.
    - The source context retrieved for this question.
    - The model's answer to the question.
    - The chat history so far (previous questions and answers).

    Your job is to compare the model's answer to BOTH the provided context and the facts already established in the chat history. If the answer is consistent with either the context or the chat history, it is a success. If the answer contradicts known facts or invents information, it is a hallucination. If the model refuses to answer, it is a refusal.

    USER QUESTION: {query}
    SOURCE CONTEXT: {context_text}
    CHAT HISTORY: {str(store.get('user1', []))}
    RAG ANSWER: {response}
    OUTPUT FORMAT: [Score] Justification: ...
    """
    try:
        eval_response = llm.invoke(eval_prompt)
        return eval_response.content
    except Exception as e:
        return f"Eval Error: {e}"

# --- Chain Setup ---

prompt = ChatPromptTemplate.from_messages([
        ("system", """### Role
        You are a precise Technical Knowledge Assistant. 

        ### Task
        Answer the question using ONLY the provided context. 

        ### Guidelines
        * **If context is provided:**
            - Strictly answer using only the provided context.
            - If the answer is not explicitly stated in the context, respond with: \"I'm sorry, but the provided context does not contain enough information to answer this.\"
            - Do not use outside knowledge or previous training data.
            - Provide a direct answer. Use bullet points for lists.
            - Quote the specific phrase from the context that supports your answer.
        * **If NO context is provided:**
            - Answer the question as best as you can, but you MUST clearly state in your response: \"Note: No context was provided.\"

        ### Context
        {context_text}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()
store = {}

def get_session_history(session_id: str):
    if session_id not in store: store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

llm_with_chat_history = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="question", history_messages_key="chat_history",
)

# --- Chat Logic ---

def rag_chat(query, history, score_threshold):
    global current_vector_store

    if current_vector_store is None:
        return "⚠️ Please upload a PDF first.", "N/A"
    if not query.strip():
        return "Please enter a valid question.", "N/A"

    try:
        results = current_vector_store.similarity_search_with_relevance_scores(query, k=5, score_threshold=score_threshold)
        
        if not results:
            context_text = "No relevant context found."
        else:
            context_text = "\n\n---\n\n".join([
                f"Page: {doc[0].metadata.get('page', 'Unknown')}\n"
                f"Content:\n{doc[0].page_content}"
                for doc in results
            ])

        response = llm_with_chat_history.invoke(
            {"question": query, "context_text": context_text},
            config={"configurable": {"session_id": "user1"}}
        )
        
        eval_res = get_eval(query, context_text, response)
        return response, eval_res

    except Exception as e:
        return f"Error: {str(e)}", "Error"

# --- Gradio UI ---


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Nvidia RAG Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            upload_status = gr.Textbox(label="Status", value="Waiting...", interactive=False)
            score_slider = gr.Slider(label="Score Threshold", minimum=0.0, maximum=1.0, value=0.2, step=0.01)
        
        with gr.Column(scale=2):
            # Explicitly set type="messages" to match the format we are sending
            chatbot = gr.Chatbot(height=500)
            query_box = gr.Textbox(label="Your Question")
            send_btn = gr.Button("Send", variant="primary")
    
    with gr.Row():
        eval_box = gr.Textbox(label="Evaluation", interactive=False)

    file_input.upload(fn=process_uploaded_pdf, inputs=[file_input], outputs=[upload_status])

    def user_chat(user_message, history, score_threshold):
        if not user_message.strip():
            return history, "", "Please enter a valid question."
        if history is None: history = []
        
        rag_response, eval_res = rag_chat(user_message, history, score_threshold)
        # --- FIX: Use Dictionary Format ---
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": rag_response})
        
        return history, "", eval_res

    send_btn.click(user_chat, inputs=[query_box, chatbot, score_slider], outputs=[chatbot, query_box, eval_box])
    query_box.submit(user_chat, inputs=[query_box, chatbot, score_slider], outputs=[chatbot, query_box, eval_box])

if __name__ == "__main__":
    demo.launch()