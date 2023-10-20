from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from envs.tabular_world import TabularWorld, GameState, get_next_game_state


def construct_mapping(world: TabularWorld) -> Dict[int, GameState]:
    """Construct a mapping from tabular states to observations.

    Args:
        world: TabularWorld object.

    Returns:
        mapping: Mapping from tabular states to observations.
    """
    current_tabular_state = 0
    total_num_states = world.num_states
    to_visit: Set[int] = {current_tabular_state}
    visited: Set[int] = set()
    mapping: Dict[int, GameState] = {}

    pbar = tqdm(total=total_num_states, desc="Constructing mapping", unit="state")
    with pbar:
        while len(to_visit) > 0:
            current_tabular_state = to_visit.pop()
            visited.add(current_tabular_state)

            if len(mapping) == 0:
                # First state
                # We assume it's 0, 0, 0, 0
                mapping[current_tabular_state] = GameState(x=0, y=0, dir=0, object=0)

            # Otherwise, the mapping should already be present
            game_state: GameState = mapping[current_tabular_state]

            # Get the next states
            next_tabular_states: torch.Tensor = world.transitions[
                current_tabular_state, :
            ]
            for action in range(world.num_actions):
                next_tabular_state = next_tabular_states[action].item()
                if (
                    next_tabular_state >= 0 and next_tabular_state not in visited
                ):  # valid state that has not been visited
                    to_visit.add(next_tabular_state)
                    mapping[next_tabular_state] = get_next_game_state(
                        current_tabular_state=current_tabular_state,
                        next_tabular_state=next_tabular_state,
                        current_game_state=game_state,
                        action=action,
                    )
            pbar.update(1)
        pbar.close()

    return mapping


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
    progress: bool = True,
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
    for tabular_state in tqdm(mapping, desc="Making grid", disable=not progress):
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
