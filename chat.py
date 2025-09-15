import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import gradio as gr

# Load environment variables
load_dotenv()

# ========== CONFIG ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = "gpt-4o-mini"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "universal-chatbot")
TOP_K = 10
# ============================

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return response.data[0].embedding


def chat_with_pdfs(query):
    # Get query embedding and search Pinecone
    query_vec = get_embedding(query)
    results = index.query(vector=query_vec, top_k=TOP_K, include_metadata=True)

    # Build context from retrieved chunks
    context_lines = []
    for match in results.matches:
        text = match.metadata.get("text", "")
        source = match.metadata.get("source", "unknown")
        page = match.metadata.get("page", "?")
        context_lines.append(f"[{source} - Page {page}]: {text}")

    context = "\n\n".join(context_lines)
    
    # Generate answer with GPT
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer questions based on the provided PDF content from UK planning documents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
    )
    return response.choices[0].message.content


# Launch Gradio interface
gr.Interface(
    fn=chat_with_pdfs,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the Universal planning documents..."),
    outputs=gr.Textbox(lines=20),
    title="Universal Planning Documents RAG Chatbot",
    description="Ask questions about Universal's Entertainment Resort Complex planning application in Bedford."
).launch()
