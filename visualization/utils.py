from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from envs.tabular_world import GameState, ACTION_FORWARD


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
