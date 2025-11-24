from typing import Dict, List
import numpy as np

def q_learning_policy(state_key: str, action_space: List[str], q_table: Dict, epsilon: float = 0.1) -> str:
    if np.random.random() < epsilon:
        return np.random.choice(action_space)  # Explore
    else:
        # Exploit: choose action with highest Q-value
        q_values = [q_table.get((state_key, action), 0) for action in action_space]
        return action_space[np.argmax(q_values)]


def encode_state(state: dict) -> str:
    context_count = len(state["context"])
    has_rewritten = state["current_query"] != state["original_query"]
    prev_reward = max(state["previous_rewards"]) if state["previous_rewards"] else 0
    return f"ctx_{context_count}_rewritten_{has_rewritten}_reward_{prev_reward:.1f}"


def update_q_table(q_table: Dict, state: str, action: str, reward: float, 
                   next_state: str, action_space: List[str], learning_rate: float = 0.1, 
                   discount: float = 0.9):
    current_q = q_table.get((state, action), 0)
    max_next_q = max([q_table.get((next_state, a), 0) for a in action_space])
    q_table[(state, action)] = current_q + learning_rate * (reward + discount * max_next_q - current_q)
