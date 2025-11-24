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

# Function to initialize training parameters
def initialize_training_params() -> Dict[str, Union[float, int]]:
    """
    Initialize training parameters such as learning rate, number of episodes, and discount factor.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing the initialized training parameters.
    """
    params = {
        "learning_rate": 0.01,  # Learning rate for policy updates
        "num_episodes": 100,   # Total number of training episodes
        "discount_factor": 0.99  # Discount factor for future rewards
    }
    return params

# Function to update policy based on reward
def update_policy(
    policy: Dict[str, Dict[str, Union[float, str]]], 
    state: Dict[str, object], 
    action: str, 
    reward: float, 
    learning_rate: float
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Update the policy based on the reward received.

    Args:
        policy (Dict[str, Dict[str, Union[float, str]]]): The current policy to be updated.
        state (Dict[str, object]): The current state of the environment.
        action (str): The action taken by the agent.
        reward (float): The reward received for the action.
        learning_rate (float): The learning rate for updating the policy.

    Returns:
        Dict[str, Dict[str, Union[float, str]]]: The updated policy.
    """
    # Example: Simple policy update (to be replaced with a proper RL algorithm)
    policy[state["query"]] = {
        "action": action,  # Store the action taken
        "reward": reward   # Store the reward received
    }
    return policy

# Function to track training progress
def track_progress(
    episode: int, 
    reward: float, 
    rewards_history: List[float]
) -> List[float]:
    """
    Track the training progress by storing rewards for each episode.

    Args:
        episode (int): The current episode number.
        reward (float): The reward received in the current episode.
        rewards_history (List[float]): A list to store the rewards for all episodes.

    Returns:
        List[float]: The updated rewards history.
    """
    # Append the current reward to the rewards history
    rewards_history.append(reward)
    
    # Print progress every 10 episodes
    print(f"Episode {episode}: Reward = {reward}")
    
    return rewards_history

# Function to define a policy network to select an action based on the state
def policy_network(
    state: dict, 
    action_space: List[str], 
    epsilon: float = 0.2
) -> str:
    """
    Define a policy network to select an action based on the current state using an epsilon-greedy strategy.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        epsilon (float): The probability of choosing a random action for exploration. Default is 0.2.

    Returns:
        str: The selected action from the action space.
    """
    # Use epsilon-greedy strategy: random exploration vs. exploitation
    if np.random.random() < epsilon:
        # Exploration: randomly select an action from the action space
        action = np.random.choice(action_space)
    else:
        # Exploitation: select the best action based on the current state using a simple heuristic

        # If there are no previous responses, prioritize rewriting the query
        if len(state["previous_responses"]) == 0:
            action = "rewrite_query"
        # If there are previous responses but the rewards are low, try expanding the context
        elif state["previous_rewards"] and max(state["previous_rewards"]) < 0.7:
            action = "expand_context"
        # If the context has too many chunks, try filtering the context
        elif len(state["context"]) > 5:
            action = "filter_context"
        # Otherwise, generate a response
        else:
            action = "generate_response"
    
    return action