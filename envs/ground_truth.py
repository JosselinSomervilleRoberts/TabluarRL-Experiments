from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import Dict

from algorithms.value_iteration import run_value_iteration
from envs.tabular_world import TabularWorld, GameState
from envs.mapping import (
    construct_mapping,
    find_grid_limits,
    shift_mapping,
    construct_value_grid_from_tabular,
)
from envs.mdp_utils import get_sparse_mdp, load_mdp_from_npz
from visualization.utils import draw_optimal_path, draw_policy
from scripts.utils import make_dir_recursively_if_not_exists


@dataclass
class GroundTruth:
    # Mapping from tabular states to game states
    mapping: Dict[int, GameState]

    # Value function
    V: np.ndarray  # (num_states,)

    # Q function
    Q: np.ndarray  # (num_states, num_actions)

    # Grid of values
    V_grid: np.ndarray  # (width, height)

    # Grid of Q values
    Q_grid: np.ndarray  # (width, height, num_actions)

    # Grid of counts (number of tabular states that map to each grid cell)
    count_grid: np.ndarray  # (width, height)

    # Transition matrix
    transitions: np.ndarray  # (num_states, num_actions)

    # Rewards vector
    rewards: np.ndarray  # (num_states, num_actions)

    @property
    def width(self) -> int:
        return self.V_grid.shape[0]

    @property
    def height(self) -> int:
        return self.V_grid.shape[1]


def compute_ground_truth(
    env_name: str, data_dir: str, horizon: int = 100, gamma: float = 0.95
) -> GroundTruth:
    # Load the MDP
    file_name = f"{data_dir}/{env_name}/consolidated.npz"
    world = TabularWorld(file_name, num_worlds=1, device="cpu")
    transitions, rewards = load_mdp_from_npz(file_name)

    # Run value iteration
    sparse_transitions, rewards_vector = get_sparse_mdp(transitions, rewards)
    vi = run_value_iteration(
        sparse_transitions, rewards_vector, horizon=horizon, gamma=gamma
    )
    Q = vi.optimal_qs[0]
    V = vi.optimal_values[0]

    # Construct mapping
    mapping: Dict[str, GameState] = construct_mapping(world)
    assert len(mapping) == world.num_states, "Mapping is not complete!"

    # Figure out the dimensions of the grid
    xmin, xmax, ymin, ymax = find_grid_limits(mapping)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    mapping = shift_mapping(mapping, xmin, ymin)

    # Make a grid of the values
    V_grid, count_grid = construct_value_grid_from_tabular(V, mapping, width, height)
    Q_grid, _ = construct_value_grid_from_tabular(Q, mapping, width, height)

    return GroundTruth(
        mapping=mapping,
        V=V,
        Q=Q,
        V_grid=V_grid,
        Q_grid=Q_grid,
        count_grid=count_grid,
        transitions=transitions,
        rewards=rewards,
    )


def save_ground_truth(env_name: str, gt: GroundTruth):
    """Save the ground truth to disk.

    Args:
        env_name: Name of the environment.
        transitions: Transition matrix.
        gt: Ground truth.
    """
    print("Saving ground truth...")

    # Save directory
    save_dir = f"output/envs/{env_name}"
    make_dir_recursively_if_not_exists(save_dir)

    # Save the MDP
    np.savez(f"{save_dir}/transitions.npz", gt.transitions)
    np.savez(f"{save_dir}/rewards.npz", gt.rewards)

    # Save Q and V values
    np.savez(f"{save_dir}/q.npz", gt.Q)
    np.savez(f"{save_dir}/v.npz", gt.V)

    # Save mapping
    with open(f"{save_dir}/mapping.pkl", "wb") as f:
        pickle.dump(gt.mapping, f)

    # Make a grid of the values
    np.savez(f"{save_dir}/v_grid.npz", gt.V_grid)
    np.savez(f"{save_dir}/count_grid.npz", gt.count_grid)
    np.savez(f"{save_dir}/q_grid.npz", gt.Q_grid)

    # Plot the count
    plt.figure(figsize=(20, 16))
    plt.imshow(gt.count_grid.T)
    plt.colorbar()
    plt.savefig(f"{save_dir}/count_grid.png")

    # Plot the grid
    plt.figure(figsize=(20, 16))
    plt.imshow(gt.V_grid.T)
    plt.colorbar()
    plt.savefig(f"{save_dir}/v_grid.png")
    draw_optimal_path(mapping=gt.mapping, transitions=gt.transitions, Q=gt.Q)
    plt.savefig(f"{save_dir}/path.png")

    # Make a new figure
    plt.figure(figsize=(20, 16))
    plt.imshow(gt.V_grid.T)
    draw_policy(mapping=gt.mapping, Q=gt.Q)
    plt.savefig(f"{save_dir}/policy.png")

    print("Done!")


def load_ground_truth(env_name: str) -> GroundTruth:
    """Load the ground truth from disk.

    Args:
        env_name: Name of the environment.

    Returns:
        Ground truth.
    """
    # Save directory
    save_dir = f"output/envs/{env_name}"
    if not os.path.exists(save_dir):
        raise ValueError(f"Ground truth for {env_name} does not exist.")

    # Load the MDP
    transitions = np.load(f"{save_dir}/transitions.npz")["arr_0"]
    rewards = np.load(f"{save_dir}/rewards.npz")["arr_0"]

    # Load Q and V values
    Q = np.load(f"{save_dir}/q.npz")["arr_0"]
    V = np.load(f"{save_dir}/v.npz")["arr_0"]

    # Load mapping
    with open(f"{save_dir}/mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

    # Load grids
    V_grid = np.load(f"{save_dir}/v_grid.npz")["arr_0"]
    count_grid = np.load(f"{save_dir}/count_grid.npz")["arr_0"]
    Q_grid = np.load(f"{save_dir}/q_grid.npz")["arr_0"]

    return GroundTruth(
        mapping=mapping,
        V=V,
        Q=Q,
        V_grid=V_grid,
        Q_grid=Q_grid,
        count_grid=count_grid,
        transitions=transitions,
        rewards=rewards,
    )
