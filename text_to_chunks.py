import os
import tiktoken

# ========== CONFIG ==========
TEXT_FOLDER = "text"  # folder with text files
CHUNKS_FOLDER = "chunks"  # folder for chunk files
CHUNK_SIZE = 500  # tokens per chunk
# ============================

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# ----------- Step 1: Read text from files -----------
def read_text_files():
    pages = []
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(TEXT_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    # Extract source PDF name and page number from filename
                    # Format: {pdf_name}_page_{page_num}.txt
                    parts = filename.replace(".txt", "").split("_page_")
                    source = parts[0] + ".pdf"
                    page_num = int(parts[1])
                    pages.append({"text": text, "source": source, "page": page_num})
    return pages


# ----------- Step 2: Chunk text -----------
def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

# ----------- Step 3: Process text files and write chunks -----------
def process_text_to_chunks():
    # Create chunks folder if it doesn't exist
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)
    
    pages = read_text_files()
    total_chunks = 0
    
    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk_num, chunk in enumerate(chunks):
            chunk_filename = f"{page['source'].replace('.pdf', '')}_page_{page['page']}_chunk_{chunk_num}.txt"
            chunk_filepath = os.path.join(CHUNKS_FOLDER, chunk_filename)
            
            with open(chunk_filepath, "w", encoding="utf-8") as f:
                f.write(chunk)
            
            total_chunks += 1
    
    print(f"Created {total_chunks} chunk files in {CHUNKS_FOLDER} folder.")

process_text_to_chunks()