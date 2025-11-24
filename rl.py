from typing import Dict, List, Union
from rag import generate_embeddings
import numpy as np

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors, ranging from -1 to 1.
    """
    # Compute the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    # Compute the magnitude (norm) of the first vector
    norm_vec1 = np.linalg.norm(vec1)
    # Compute the magnitude (norm) of the second vector
    norm_vec2 = np.linalg.norm(vec2)
    # Return the cosine similarity as the ratio of the dot product to the product of the norms
    return dot_product / (norm_vec1 * norm_vec2)

def define_state(
    query: str, 
    context_chunks: List[str], 
    rewritten_query: str = None, 
    previous_responses: List[str] = None, 
    previous_rewards: List[float] = None
) -> dict:
    """
    Define the state representation for the reinforcement learning agent.
    
    Args:
        query (str): The original user query.
        context_chunks (List[str]): Retrieved context chunks from the knowledge base.
        rewritten_query (str, optional): A reformulated version of the original query.
        previous_responses (List[str], optional): List of previously generated responses.
        previous_rewards (List[float], optional): List of rewards received for previous actions.
    
    Returns:
        dict: A dictionary representing the current state with all relevant information.
    """
    state = {
        "original_query": query,                                    # The initial query from the user
        "current_query": rewritten_query if rewritten_query else query,  # Current version of the query (may be rewritten)
        "context": context_chunks,                                 # Retrieved context chunks from the knowledge base
        "previous_responses": previous_responses if previous_responses else [],  # History of generated responses
        "previous_rewards": previous_rewards if previous_rewards else []         # History of received rewards
    }
    return state

def define_action_space() -> List[str]:
    """
    Define the set of possible actions the reinforcement learning agent can take.
    
    Actions include:
    - rewrite_query: Reformulate the original query to improve retrieval
    - expand_context: Retrieve additional context chunks
    - filter_context: Remove irrelevant context chunks
    - generate_response: Generate a response based on current query and context
    
    Returns:
        List[str]: A list of available actions.
    """

    # Define the set of actions the agent can take
    actions = ["rewrite_query", "expand_context", "filter_context", "generate_response"]
    return actions

def calculate_reward(response: str, ground_truth: str) -> float:
    """
    Calculate a reward value by comparing the generated response to the ground truth.
    
    Uses cosine similarity between the embeddings of the response and ground truth
    to determine how close the response is to the expected answer.
    
    Args:
        response (str): The generated response from the RAG pipeline.
        ground_truth (str): The expected correct answer.
    
    Returns:
        float: A reward value between -1 and 1, where higher values indicate 
               greater similarity to the ground truth.
    """
    # Generate embeddings for both the response and ground truth
    response_embedding = generate_embeddings([response])[0]
    ground_truth_embedding = generate_embeddings([ground_truth])[0]
    
    # Calculate cosine similarity between the embeddings as the reward
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)
    return similarity