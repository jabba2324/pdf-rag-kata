# Function to implement the training loop
from typing import Dict, List, Optional, Tuple, Union

from rag import basic_rag_pipeline, retrieve_relevant_chunks
from rl import calculate_reward, define_action_space, define_state, initialize_training_params
from rl_step import rl_step

def training_loop(
    query_text: str, 
    ground_truth: str, 
    params: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[Dict[str, Dict[str, Union[float, str]]], List[float], List[List[str]], Optional[str]]:
    """
    Implement the training loop for RL-enhanced RAG.

    Args:
        query_text (str): The input query text for the RAG pipeline.
        ground_truth (str): The expected correct answer for the query.
        params (Optional[Dict[str, Union[float, int]]]): Training parameters such as learning rate, 
            number of episodes, and discount factor. If None, default parameters are initialized.

    Returns:
        Tuple: A tuple containing:
            - policy (Dict[str, Dict[str, Union[float, str]]]): The updated policy after training.
            - rewards_history (List[float]): A list of rewards received in each episode.
            - actions_history (List[List[str]]): A list of actions taken in each episode.
            - best_response (Optional[str]): The best response generated during training.
    """
    # Initialize training parameters if not provided
    if params is None:
        params = initialize_training_params()
    
    # Initialize variables to track progress
    rewards_history: List[float] = []  # List to store rewards for each episode
    actions_history: List[List[str]] = []  # List to store actions taken in each episode
    policy: Dict[str, Dict[str, Union[float, str]]] = {}  # Policy dictionary to store actions and rewards
    action_space: List[str] = define_action_space()  # Define the action space
    best_response: Optional[str] = None  # Variable to store the best response
    best_reward: float = -1  # Initialize the best reward to a very low value
    
       # Get initial performance from the simple RAG pipeline for comparison
    simple_response: str = basic_rag_pipeline(query_text)
    simple_reward: float = calculate_reward(simple_response, ground_truth)
    print(f"Simple RAG reward: {simple_reward:.4f}")

    # Start the training loop
    for episode in range(params["num_episodes"]):
        # Reset the environment with the same query

        context_chunks: List[str] = retrieve_relevant_chunks(query_text)
        state: Dict[str, object] = define_state(query_text, context_chunks)
        episode_reward: float = 0  # Initialize the reward for the current episode
        episode_actions: List[str] = []  # Initialize the list of actions for the current episode
        
        # Maximum number of steps per episode to prevent infinite loops
        for step in range(10):
            # Perform a single RL step
            state, action, reward, response = rl_step(state, action_space, ground_truth)
            episode_actions.append(action)  # Record the action taken
            
            # If a response is generated, end the episode
            if response:
                episode_reward = reward  # Update the episode reward
                
                # Track the best response and reward
                if reward > best_reward:
                    best_reward = reward
                    best_response = response
                
                break  # Exit the loop as the episode ends
        
        # Update rewards and actions history
        rewards_history.append(episode_reward)
        actions_history.append(episode_actions)
        
        # Print progress every 5 episodes
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.4f}, Actions = {episode_actions}")
  # Compare the best RL-enhanced RAG reward with the simple RAG reward
    improvement: float = best_reward - simple_reward
    print(f"\nTraining completed:")
    print(f"Simple RAG reward: {simple_reward:.4f}")
    print(f"Best RL-enhanced RAG reward: {best_reward:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")

    return policy, rewards_history, actions_history, best_response