import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

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

def generate_embeddings(text):
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    embeddings = [item.embedding for item in response.data]
    return embeddings

def retrieve_relevant_chunks(query, top_k=TOP_K):
    query_vec = generate_embeddings(query)[0]
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    context_lines = []
    for match in results.matches:
        text = match.metadata.get("text", "")
        source = match.metadata.get("source", "unknown")
        page = match.metadata.get("page", "?")
        context_lines.append(f"[{source} - Page {page}]: {text}")
    return context_lines


def construct_prompt(query, context):
    # Construct a prompt using the current query and context
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return prompt

def generate_response(prompt):
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer questions based on the provided PDF content from UK planning documents."},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content

def basic_rag_pipeline(query):
    # Build context from retrieved chunks
    context_lines = retrieve_relevant_chunks(query)

    context = "\n\n".join(context_lines)
    
    prompt = construct_prompt(query, context)

    # Generate answer with GPT
    response = generate_response(prompt)
    return response



