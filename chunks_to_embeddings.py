import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# ========== CONFIG ==========
CHUNKS_FOLDER = "chunks"  # folder with chunk files
EMBEDDINGS_FOLDER = "embeddings"  # folder for embedding files
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ============================

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------- Step 1: Generate embeddings -----------
def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return response.data[0].embedding

# ----------- Step 2: Process chunk files and create embeddings -----------
def process_chunks_to_embeddings():
    # Create embeddings folder if it doesn't exist
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
    
    processed_count = 0
    
    for filename in os.listdir(CHUNKS_FOLDER):
        if filename.endswith(".txt"):
            chunk_filepath = os.path.join(CHUNKS_FOLDER, filename)
            
            # Read chunk text
            with open(chunk_filepath, "r", encoding="utf-8") as f:
                chunk_text = f.read().strip()
            
            if chunk_text:
                # Generate embedding
                embedding = get_embedding(chunk_text)
                
                # Create embedding data
                embedding_data = {
                    "filename": filename,
                    "text": chunk_text,
                    "embedding": embedding
                }
                
                # Save embedding to JSON file
                embedding_filename = filename.replace(".txt", "_embedding.json")
                embedding_filepath = os.path.join(EMBEDDINGS_FOLDER, embedding_filename)
                
                with open(embedding_filepath, "w", encoding="utf-8") as f:
                    json.dump(embedding_data, f, indent=2)
                
                processed_count += 1
                print(f"Processed {processed_count}: {filename}")
    
    print(f"Created {processed_count} embedding files in {EMBEDDINGS_FOLDER} folder.")

process_chunks_to_embeddings()