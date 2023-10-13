# Description: Utility functions for loading and processing MDPs.
# The format of the MDP files is a .npz file with two arrays:
# - "transitions": A (num_states, num_actions) array of integers, where each entry is the next state.
# - "rewards": A (num_states, num_actions) array of floats, where each entry is the reward.
# You can use the Effective Horizon dataset to find MDPs in this format.
# Source code from the Effective Hoizon repo.
# Link: https://github.com/cassidylaidlaw/effective-horizon


from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix


def load_mdp_from_npz(mdp_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an MDP from a .npz file.

    Args:
        mdp_path: Path to the .npz file.

    Returns:
        transitions: A (num_states, num_actions) array of integers, where each entry is the next state.
        rewards: A (num_states, num_actions) array of floats, where each entry is the reward.
    """
    mdp = np.load(mdp_path)

    transitions = mdp["transitions"]
    num_states, num_actions = transitions.shape
    done_state = num_states
    num_states += 1
    transitions = np.concatenate(
        [transitions, np.zeros((1, num_actions), dtype=transitions.dtype)]
    )
    transitions[transitions == -1] = done_state
    transitions[done_state, :] = done_state
    rewards = np.concatenate([mdp["rewards"], np.zeros((1, num_actions))])

    return transitions, rewards


def get_sparse_mdp(
    transitions: np.ndarray, rewards: np.ndarray
) -> Tuple[csr_matrix, np.ndarray]:
    """Convert an MDP to a sparse representation.

    Args:
        transitions: A (num_states, num_actions) array of integers, where each entry is the next state.
        rewards: A (num_states, num_actions) array of floats, where each entry is the reward.

    Returns:
        sparse_transitions: A (num_state_actions, num_states) sparse matrix, where each row is a state-action pair.
        rewards_vector: A (num_state_actions,) vector of rewards.
    """
    num_states, num_actions = transitions.shape
    num_state_actions = num_states * num_actions
    sparse_transitions = csr_matrix(
        (
            np.ones(num_state_actions),
            (np.arange(num_state_actions, dtype=int), transitions.ravel()),
        ),
        shape=(num_state_actions, num_states),
        dtype=np.float32,
    )
    rewards_vector = rewards.ravel().astype(np.float32)
    return sparse_transitions, rewards_vector
