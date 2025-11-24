# Function to rewrite the query for better document retrieval
import os
from typing import List
from rag import OPENAI_CHAT_MODEL, generate_embeddings, retrieve_relevant_chunks
from rl import cosine_similarity
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def rewrite_query(
    query: str, 
    context_chunks: List[str], 
    model: str = OPENAI_CHAT_MODEL, 
    max_tokens: int = 100, 
    temperature: float = 0.3
) -> str:
    """
    Use the LLM to rewrite the query for better document retrieval.

    Args:
        query (str): The original query text.
        context_chunks (List[str]): A list of context chunks retrieved so far.
        model (str): The model to use for generating the rewritten query. Default is "google/gemma-2-2b-it".
        max_tokens (int): Maximum number of tokens in the rewritten query. Default is 100.
        temperature (float): Sampling temperature for response diversity. Default is 0.3.

    Returns:
        str: The rewritten query optimized for document retrieval.
    """
    # Construct a prompt for the LLM to rewrite the query
    rewrite_prompt = f"""
    You are a query optimization assistant. Your task is to rewrite the given query to make it more effective 
    for retrieving relevant information. The query will be used for document retrieval.
    
    Original query: {query}
    
    Based on the context retrieved so far:
    {' '.join(context_chunks[:2]) if context_chunks else 'No context available yet'}
    
    Rewrite the query to be more specific and targeted to retrieve better information.
    Rewritten query:
    """
    
    # Use the LLM to generate a rewritten query
    response = client.chat.completions.create(
        model=model, # Specify the model to use for generating the response
        max_tokens=max_tokens, # Maximum number of tokens in the response
        temperature=temperature, # Sampling temperature for response diversity
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )
    
    # Extract and return the rewritten query from the response
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

# Function to expand the context by retrieving additional chunks
def expand_context(query: str, current_chunks: List[str], top_k: int = 3) -> List[str]:
    """
    Expand the context by retrieving additional chunks.

    Args:
        query (str): The query text for which additional context is needed.
        current_chunks (List[str]): The current list of context chunks.
        top_k (int): The number of additional chunks to retrieve. Default is 3.

    Returns:
        List[str]: The expanded list of context chunks including new unique chunks.
    """
    # Retrieve more chunks than currently available
    additional_chunks = retrieve_relevant_chunks(query, top_k=top_k + len(current_chunks))
    
    # Filter out chunks that are already in the current context
    new_chunks = []
    for chunk in additional_chunks:
        if chunk not in current_chunks:
            new_chunks.append(chunk)
    
    # Add new unique chunks to the current context, limited to top_k
    expanded_context = current_chunks + new_chunks[:top_k]
    return expanded_context

# Function to filter the context to keep only the most relevant chunks
def filter_context(query: str, context_chunks: List[str]) -> List[str]:
    """
    Filter the context to keep only the most relevant chunks.

    Args:
        query (str): The query text for which relevance is calculated.
        context_chunks (List[str]): The list of context chunks to filter.

    Returns:
        List[str]: A filtered list of the most relevant context chunks.
    """
    if not context_chunks:
        return []
        
    # Generate embeddings for the query and each chunk
    query_embedding = generate_embeddings(query)[0]
    chunk_embeddings = [generate_embeddings(chunk)[0] for chunk in context_chunks]
    
    # Calculate relevance scores for each chunk
    relevance_scores = []
    for chunk_embedding in chunk_embeddings:
        score = cosine_similarity(query_embedding, chunk_embedding)
        relevance_scores.append(score)
    
    # Sort chunks by relevance scores in descending order
    sorted_chunks = [x for _, x in sorted(zip(relevance_scores, context_chunks), reverse=True)]
    
    # Keep the top 5 most relevant chunks or fewer if less than 5 are available
    filtered_chunks = sorted_chunks[:min(5, len(sorted_chunks))]
    
    return filtered_chunks