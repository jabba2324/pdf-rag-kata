# Function to perform a single RL step
from typing import List
from q import encode_state, q_learning_policy, update_q_table
from rag import construct_prompt, generate_response, retrieve_relevant_chunks
from rl import calculate_reward, policy_network
from rl_actions import expand_context, filter_context, rewrite_query


def rl_step(
    state: dict, 
    action_space: List[str], 
    ground_truth: str,
    q_table,
    epsilon: float = 0.5
) -> tuple[dict, str, float, str]:
    """
    Perform a single RL step: select an action, execute it, and calculate the reward.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        ground_truth (str): The expected correct answer to calculate the reward.

    Returns:
        tuple: A tuple containing:
            - state (dict): The updated state after executing the action.
            - action (str): The action selected by the policy network.
            - reward (float): The reward received for the action.
            - response (str): The response generated (if applicable).
    """
    # Select an action using the policy network
    current_state_key = encode_state(state)
    action: str = q_learning_policy(current_state_key, action_space, q_table, epsilon)
    response: str = None  # Initialize response as None
    reward: float = 0  # Initialize reward as 0

    # Execute the selected action
    if action == "rewrite_query":
        # Rewrite the query to improve retrieval
        rewritten_query: str = rewrite_query(state["original_query"], state["context"])
        state["current_query"] = rewritten_query  # Update the current query in the state
        # Retrieve new context based on the rewritten query
        new_context: List[str] = retrieve_relevant_chunks(rewritten_query)
        state["context"] = new_context  # Update the context in the state

    elif action == "expand_context":
        # Expand the context by retrieving additional chunks
        expanded_context: List[str] = expand_context(state["current_query"], state["context"])
        state["context"] = expanded_context  # Update the context in the state

    elif action == "filter_context":
        # Filter the context to keep only the most relevant chunks
        filtered_context: List[str] = filter_context(state["current_query"], state["context"])
        state["context"] = filtered_context  # Update the context in the state

    elif action == "generate_response":
        # Construct a prompt using the current query and context
        prompt: str = construct_prompt(state["current_query"], state["context"])
        # Generate a response using the LLM
        response: str = generate_response(prompt)
        # Calculate the reward based on the similarity between the response and the ground truth
        reward: float = calculate_reward(response, ground_truth)
        # Update the state with the new response and reward
        state["previous_responses"].append(response)
        state["previous_rewards"].append(reward)

    # Return the updated state, selected action, reward, and response
    next_state_key = encode_state(state)  # After action execution
    update_q_table(q_table, current_state_key, action, reward, next_state_key, action_space)
    return state, action, reward, response