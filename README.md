# PDF to Vector Pipeline

This repository contains a pipeline for processing PDF documents into vector embeddings and storing them in Pinecone for semantic search.

## Data Source

The raw PDF data comes from the UK Government's planning application for Universal's Entertainment Resort Complex in Bedford:
https://www.gov.uk/government/collections/request-for-planning-permission-entertainment-resort-complex-bedford

## Prerequisites

1. **OpenAI API Key** - For generating embeddings
2. **Pinecone API Key** - For vector storage
3. **Python 3.12+** with required packages

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

2. Edit the `.env` file with your actual API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=universal-chatbot
```

## Usage

Run the scripts in this order:

### 1. Extract Text from PDFs
```bash
python pdf_to_text.py
```
- **Purpose**: Extracts text from PDF files and saves each page as a separate text file
- **Input**: PDF files in `universal_raw_data/` folder
- **Output**: Text files in `text/` folder (format: `{pdf_name}_page_{page_num}.txt`)

### 2. Convert Text to Chunks
```bash
python text_to_chunks.py
```
- **Purpose**: Splits text files into smaller chunks for better embedding quality
- **Input**: Text files from `text/` folder
- **Output**: Chunk files in `chunks/` folder (format: `{pdf_name}_page_{page_num}_chunk_{chunk_num}.txt`)
- **Configuration**: Adjust `CHUNK_SIZE` (default: 500 tokens)

### 3. Generate Embeddings
```bash
python chunks_to_embeddings.py
```
- **Purpose**: Creates OpenAI embeddings for each text chunk
- **Input**: Chunk files from `chunks/` folder
- **Output**: Embedding JSON files in `embeddings/` folder
- **Note**: This step makes API calls to OpenAI and may take time/cost money

### 4. Upload to Pinecone
```bash
python embeddings_to_pinecone.py
```
- **Purpose**: Uploads embeddings to Pinecone vector database
- **Input**: Embedding files from `embeddings/` folder
- **Output**: Vectors stored in Pinecone index
- **Configuration**: Set `PINECONE_INDEX_NAME` (default: "universal-chatbot")

### 5. Chat with Documents (RAG)
```bash
python start_rag.py
```
- **Purpose**: Provides a web interface to chat with your PDF documents using RAG
- **Input**: User questions via web interface
- **Output**: AI-generated answers based on document content
- **Features**: Retrieves relevant chunks from Pinecone and generates contextual answers with GPT
- **Access**: Opens web interface at `http://localhost:7860`
- **Example question**: "When is the park opening?"

### 6. Compare RAG vs RL-Enhanced RAG
```bash
python start_rl.py
```
- **Purpose**: Compares simple RAG with reinforcement learning enhanced RAG
- **Input**: Query and ground truth via web interface
- **Output**: JSON comparison of both approaches with similarity scores
- **Features**: Trains an RL agent to optimize RAG responses and compares performance
- **Access**: Opens web interface at `http://localhost:7860`
- **Example**: Query: "When is the park opening?" | Ground truth: "2031"

## Folder Structure

```
pdf-to-vector/
├── universal_raw_data/     # Original PDF files
├── text/                   # Extracted text files
├── chunks/                 # Text chunks
├── embeddings/             # Generated embeddings
├── pdf_to_text.py         # Step 1: PDF → Text
├── text_to_chunks.py      # Step 2: Text → Chunks
├── chunks_to_embeddings.py # Step 3: Chunks → Embeddings
├── embeddings_to_pinecone.py # Step 4: Embeddings → Pinecone
├── start_rag.py          # Step 5: RAG Chat Interface
├── start_rl.py           # Step 6: RAG vs RL-Enhanced RAG Comparison
├── rag.py                # RAG core functions
├── rl_vs_rag.py          # RL vs RAG comparison logic
├── rl.py                 # RL core functions
├── rl_actions.py         # RL action implementations
├── rl_loop.py            # RL training loop
├── rl_step.py            # RL step execution
├── .env.example          # API keys template
└── requirements.txt
```

## Notes

- Each script creates its output folder automatically
- The pipeline preserves metadata (source PDF, page number, chunk number)
- Vector IDs in Pinecone use the chunk filename format
- Chunk numbers are per-page (starting from 0 for each page)
- The pipeline uses OpenAI's `text-embedding-3-large` model