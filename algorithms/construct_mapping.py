from typing import Dict, Set

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
