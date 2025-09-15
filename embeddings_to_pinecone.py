import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Load environment variables
load_dotenv()

# ========== CONFIG ==========
EMBEDDINGS_FOLDER = "embeddings"  # folder with embedding files
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "universal-chatbot")
# ============================

# Try different pinecone import methods
pc = Pinecone(api_key=PINECONE_API_KEY)
# ----------- Step 1: Parse metadata from filename -----------
def parse_metadata(filename):
    # Format: {pdf_name}_page_{page_num}_chunk_{chunk_num}_embedding.json
    parts = filename.replace("_embedding.json", "").split("_")
    
    # Find page and chunk indices
    page_idx = parts.index("page") + 1
    chunk_idx = parts.index("chunk") + 1
    
    # Extract PDF name (everything before "_page_")
    pdf_name = "_".join(parts[:page_idx-1]) + ".pdf"
    page_num = int(parts[page_idx])
    chunk_num = int(parts[chunk_idx])
    
    return {"source": pdf_name, "page": page_num, "chunk": chunk_num}

# ----------- Step 2: Upload embeddings to Pinecone -----------
def embeddings_to_pinecone():
    vectors = []
    
    # Read all embedding files
    for filename in os.listdir(EMBEDDINGS_FOLDER):
        if filename.endswith("_embedding.json"):
            filepath = os.path.join(EMBEDDINGS_FOLDER, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Parse metadata from filename
            metadata = parse_metadata(filename)
            metadata["text"] = data["text"]
            
            # Create vector tuple (id, embedding, metadata)
            vector_id = filename.replace("_embedding.json", "")
            vectors.append((vector_id, data["embedding"], metadata))
    
    if not vectors:
        print("No embedding files found.")
        return
    
    # Get dimension from first embedding
    dim = len(vectors[0][1])
    
    # Create index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Upsert vectors into Pinecone one by one
    print("Uploading vectors to Pinecone...")
    for i, (vector_id, embedding, metadata) in enumerate(vectors):
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        if (i + 1) % 100 == 0:
            print(f"Uploaded {i + 1}/{len(vectors)} vectors...")
    print(f"Indexed {len(vectors)} vectors into Pinecone.")

embeddings_to_pinecone()