# Filter warnings
import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any

from algorithms.construct_mapping import construct_mapping
from algorithms.value_iteration import run_value_iteration
from envs.tabular_world import TabularWorld, GameState, ACTION_FORWARD
from envs.mdp_utils import get_sparse_mdp, load_mdp_from_npz


def make_dir_recursively_if_not_exists(path: str):
    """Make a directory recursively if it does not exist.

    Args:
        path: Path to the directory.
    """
    splits = path.split("/")
    for i in range(1, len(splits) + 1):
        dir_name = "/".join(splits[:i])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-MultiRoom-N6-v0",
        help="Name of the MDP.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Directory containing the MDPs.",
    )
    parser.add_argument(
        "--horizon", type=int, default=100, help="Horizon for value iteration."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="Discount factor for value iteration."
    )
    return parser.parse_args()


def find_grid_limits(mapping: Dict[int, GameState]) -> Tuple[int, int, int, int]:
    """Find the limits of the grid.

    Args:
        mapping: Mapping from tabular states to game states.

    Returns:
        xmin: Minimum x value.
        xmax: Maximum x value.
        ymin: Minimum y value.
        ymax: Maximum y value.
    """
    # We add a border of 1 around the grid
    xmin = min([game_state.x for game_state in mapping.values()]) - 1
    xmax = max([game_state.x for game_state in mapping.values()]) + 1
    ymin = min([game_state.y for game_state in mapping.values()]) - 1
    ymax = max([game_state.y for game_state in mapping.values()]) + 1
    return xmin, xmax, ymin, ymax


def shift_mapping(
    mapping: Dict[int, GameState], xmin: int, ymin: int
) -> Dict[int, GameState]:
    """Shift the mapping so that the minimum x and y values are 0.

    Args:
        mapping: Mapping from tabular states to game states.
        xmin: Minimum x value.
        ymin: Minimum y value.

    Returns:
        mapping: Shifted mapping.
    """
    for tabular_state in mapping:
        game_state: GameState = mapping[tabular_state]
        game_state.x -= xmin
        game_state.y -= ymin
    return mapping


def construct_value_grid_from_tabular(
    tab: np.ndarray,
    mapping: Dict[int, GameState],
    width: int,
    height: int,
    args: Optional[Dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a grid of the Q or V values from the tabular values.

    Args:
        tab: Q (num_states, num_actions) or V (num_states,) values.
        mapping: Mapping from tabular states to game states.
        width: Width of the grid.
        height: Height of the grid.
        args: Additional arguments. Can contain:
            - only_if_object_is: value -> Only plot the V values if the object is equal to value.

    Returns:
        grid: Grid of Q (width, height, num_actions) or V (width, height) values.
        count: Number of times each cell was visited.
    """
    shape = (width, height) if len(tab.shape) == 1 else (width, height, tab.shape[1])
    grid = np.zeros(shape)
    count = np.zeros((width, height))
    for tabular_state in tqdm(mapping, desc="Making grid"):
        game_state: GameState = mapping[tabular_state]
        if args is not None and "only_if_object_is" in args:
            if game_state.object != args["only_if_object_is"]:
                continue
        grid[game_state.x, game_state.y] += tab[tabular_state]
        count[game_state.x, game_state.y] += 1

    # Normalize
    if grid.shape != count.shape:
        count = np.expand_dims(count, axis=-1)
    grid = grid / np.maximum(count, 1)
    return grid, count


def plot_game_state(game_state: GameState):
    """Plots the game state as an arrow in pyplot.

    Args:
        game_state: Game state.
    """
    color = "blue" if game_state.object == 0 else "red"
    plt.arrow(
        game_state.x,
        game_state.y,
        0.2 * np.cos(game_state.dir * np.pi / 2),
        0.2 * np.sin(game_state.dir * np.pi / 2),
        color=color,
        width=0.05,
    )


def draw_optimal_path(
    mapping: Dict[int, GameState],
    transitions: np.ndarray,
    Q: np.array,
    starting_tabular_state: int = 0,
    done_tabular_state: int = -1,
):
    """Draw the optimal path using pyplot.

    Args:
        transitions: Transition matrix.
        Q: Q values.
        xmin: Minimum x value.
        ymin: Minimum y value.
        starting_tabular_state: Starting tabular state.
        done_tabular_state: Tabular state that indicates the end of the path.
    """
    if done_tabular_state == -1:
        done_tabular_state = len(Q) - 1

    # Find the optimal path
    state = starting_tabular_state
    path = [state]
    t = 0
    while state != done_tabular_state and t < 100:
        action = np.argmax(Q[state])
        state = transitions[state, action]
        path.append(state)
        t += 1

    # Plot the path
    for i in tqdm(range(len(path)), desc="Drawing path"):
        # Draw an arrow of length 0.4 from the center of the current state in the game state direction
        # The arrow should be blue if object is 0, red if object is 1
        game_state: GameState = mapping[path[i]]
        plot_game_state(game_state)


def draw_policy(mapping: Dict[int, GameState], Q: np.array):
    """At each cell of the grid, draw the the direction of the forward movement.

    Args:
        mapping: Mapping from tabular states to game states.
        Q: Q values.
    """
    for state in tqdm(mapping, desc="Drawing policy"):
        game_state: GameState = mapping[state]
        optimal_action = np.argmax(Q[state])
        if optimal_action == ACTION_FORWARD:
            plot_game_state(game_state)


def main(args: argparse.Namespace):
    # Save directory
    save_dir = f"output/envs/{args.env_name}"
    make_dir_recursively_if_not_exists(save_dir)

    # Load the MDP
    file_name = f"{args.data_dir}/{args.env_name}/consolidated.npz"
    world = TabularWorld(file_name, num_worlds=1, device="cpu")
    transitions, rewards = load_mdp_from_npz(file_name)

    # Run value iteration
    sparse_transitions, rewards_vector = get_sparse_mdp(transitions, rewards)
    vi = run_value_iteration(
        sparse_transitions, rewards_vector, horizon=100, gamma=0.96
    )
    Q = vi.optimal_qs[0]
    V = vi.optimal_values[0]
    np.savez(f"{save_dir}/q.npz", Q)
    np.savez(f"{save_dir}/v.npz", V)

    # Construct mapping
    mapping: Dict[str, GameState] = construct_mapping(world)
    assert len(mapping) == world.num_states, "Mapping is not complete!"

    # Figure out the dimensions of the grid
    xmin, xmax, ymin, ymax = find_grid_limits(mapping)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    mapping = shift_mapping(mapping, xmin, ymin)
    with open(f"{save_dir}/mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)

    # Make a grid of the values
    V_grid, count = construct_value_grid_from_tabular(V, mapping, width, height)
    np.savez(f"{save_dir}/v_grid.npz", V_grid)
    np.savez(f"{save_dir}/count_grid.npz", count)
    Q_grid, _ = construct_value_grid_from_tabular(Q, mapping, width, height)
    np.savez(f"{save_dir}/q_grid.npz", Q_grid)

    # Plot the count
    plt.figure(figsize=(20, 16))
    plt.imshow(count.T)
    plt.colorbar()
    plt.savefig(f"{save_dir}/count_grid.png")

    # Plot the grid
    plt.figure(figsize=(20, 16))
    plt.imshow(V_grid.T)
    plt.colorbar()
    plt.savefig(f"{save_dir}/v_grid.png")
    draw_optimal_path(mapping=mapping, transitions=transitions, Q=Q)
    plt.savefig(f"{save_dir}/path.png")

    # Make a new figure
    plt.figure(figsize=(20, 16))
    plt.imshow(V_grid.T)
    draw_policy(mapping=mapping, Q=Q)
    plt.savefig(f"{save_dir}/policy.png")

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
