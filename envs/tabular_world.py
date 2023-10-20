# File for simulating tabular worlds from "transition" and "reward" matrices

from dataclasses import dataclass, replace
from typing import Dict, Optional, Union

import torch

from .mdp_utils import load_mdp_from_npz

# Actions
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_PICKUP = 3
ACTION_DROP = 4
ACTION_USE = 5


@dataclass
class GameState(object):
    # x-coordinate of the agent
    x: int
    # y-coordinate of the agent
    y: int
    # 0: right, 1: down, 2: left, 3: up
    dir: int
    # 0: no object, 1: object
    object: int


def get_next_game_state(
    current_tabular_state: int,
    next_tabular_state: int,
    current_game_state: GameState,
    action: int,
) -> GameState:
    """Gets the next state given the current state and the action.

    Args:
        current_tabular_state: The current tabular state.
        next_tabular_state: The next tabular state.
        current_game_state: The current game state.
        action: The action taken.

    Returns:
        The next game state.
    """

    new_game_state = replace(current_game_state)  # Copy the game state
    if next_tabular_state == current_tabular_state:
        # No change in tabular state
        return new_game_state

    # Otherwise, something changed so the action was successful
    if action == ACTION_LEFT:
        new_game_state.dir = (new_game_state.dir - 1) % 4
    elif action == ACTION_RIGHT:
        new_game_state.dir = (new_game_state.dir + 1) % 4
    elif action == ACTION_FORWARD:
        if new_game_state.dir == 0:
            new_game_state.x += 1
        elif new_game_state.dir == 1:
            new_game_state.y += 1
        elif new_game_state.dir == 2:
            new_game_state.x -= 1
        elif new_game_state.dir == 3:
            new_game_state.y -= 1
    elif action == ACTION_PICKUP:
        new_game_state.object = 1
    elif action == ACTION_DROP:
        new_game_state.object = 0
    elif action == ACTION_USE:
        pass
    else:
        raise ValueError("Invalid action")
    return new_game_state


class GameStateTracker:
    """This is a simplified version of the GameState of MiniGrid.

    It is used to map a tabular state and an actual game state"""

    def __init__(self, num_worlds: int, device: str, initial_game_state: GameState):
        self.num_worlds = num_worlds
        self.device = device
        self.initial_game_state = initial_game_state
        self.x = torch.zeros((num_worlds, 1), dtype=torch.int32, device=device)
        self.y = torch.zeros((num_worlds, 1), dtype=torch.int32, device=device)
        self.dir = torch.zeros((num_worlds, 1), dtype=torch.int32, device=device)
        self.object = torch.zeros((num_worlds, 1), dtype=torch.int32, device=device)
        self.reset()

    def reset(
        self,
        mask: Optional[torch.Tensor] = None,
        initial_game_state: Optional[GameState] = None,
    ):
        """Resets the game state tracker.

        Args:
            mask: The mask of the worlds to reset. If None, resets all worlds.
            initial_game_state: The initial game state. If None, reset to the initial game state.
                specified in the constructor.
        """
        if mask is None:
            mask = torch.ones(
                (self.num_worlds, 1), dtype=torch.int32, device=self.device
            )
        if initial_game_state is None:
            initial_game_state = self.initial_game_state

        self.x[mask] = initial_game_state.x
        self.y[mask] = initial_game_state.y
        self.dir[mask] = initial_game_state.dir
        self.object[mask] = initial_game_state.object

    def step(
        self,
        actions: torch.Tensor,
        previous_tabular_state: torch.Tensor,
        new_tabular_state: torch.Tensor,
    ):
        """Updates the game state based on the actions taken and the tabular states.

        Args:
            actions: The actions taken. (num_worlds, 1)
            previous_tabular_state: The previous tabular state. (num_worlds, 1)
            new_tabular_state: The new tabular state. (num_worlds, 1)
        """
        mask_changed = (previous_tabular_state != new_tabular_state).int()

        # Turn left
        self.dir[mask_changed & (actions == self.ACTION_LEFT)] = (self.dir - 1) % 4

        # Turn right
        self.dir[mask_changed & (actions == self.ACTION_RIGHT)] = (self.dir + 1) % 4

        # Move forward
        self.x[
            mask_changed & (actions == self.ACTION_FORWARD) & (self.dir == 0)
        ] += 1  # Heading right
        self.y[
            mask_changed & (actions == self.ACTION_FORWARD) & (self.dir == 1)
        ] += 1  # Heading down
        self.x[
            mask_changed & (actions == self.ACTION_FORWARD) & (self.dir == 2)
        ] -= 1  # Heading left
        self.y[
            mask_changed & (actions == self.ACTION_FORWARD) & (self.dir == 3)
        ] -= 1  # Heading up

        # Pick up an object
        self.object[
            mask_changed & (actions == self.ACTION_PICKUP)
        ] = 1  # has picked up an object

        # Drop an object
        self.object[
            mask_changed & (actions == self.ACTION_DROP)
        ] = 0  # has dropped an object

        # Use an object
        # Does not change our current limited version of the game state


class TabularStateToGameStateMapper:
    MAPPING: Dict[int, GameState] = {}

    def add_to_mapping(
        self, tabular_state: torch.Tensor, game_state_tracker: GameStateTracker
    ):
        """Adds a mapping from tabular state to game state.

        Args:
            tabular_state: The tabular state. (num_worlds, 1)
            game_state_tracker: The game state environment.
        """

        # Efficiently add each mapping if not already in the mapping
        for i in range(game_state_tracker.num_worlds):
            if tabular_state[i, 0].item() not in self.MAPPING:
                self.MAPPING[tabular_state[i, 0].item()] = GameState(
                    x=game_state_tracker.x[i, 0].item(),
                    y=game_state_tracker.y[i, 0].item(),
                    dir=game_state_tracker.dir[i, 0].item(),
                    object=game_state_tracker.object[i, 0].item(),
                )

    def get_game_state(self, tabular_state: int) -> Union[GameState, None]:
        """Gets the game state corresponding to the tabular state.

        Args:
            tabular_state: The tabular state. (num_worlds, 1)

        Returns:
            The game state corresponding to the tabular state if it exists, else None.
        """
        try:
            return self.MAPPING[tabular_state]
        except KeyError:
            return None


class TabularWorld:
    def __init__(self, filename: str, num_worlds: int, device: str):
        self.name = filename.split("/")[-2]
        self.num_worlds = num_worlds
        self.device = device

        transitions, rewards = load_mdp_from_npz(filename)
        # Core transition matrix
        self.transitions = torch.tensor(transitions, device=device)
        self.transition_rewards = torch.tensor(rewards, device=device)

        # Current state
        self.observations = torch.zeros((num_worlds,), dtype=torch.int32, device=device)
        self.actions = torch.zeros((num_worlds,), dtype=torch.int32, device=device)
        self.dones = torch.zeros((num_worlds,), dtype=torch.int32, device=device)
        self.rewards = torch.zeros((num_worlds,), device=device)

        # Flag for reset per world
        self.force_reset = torch.zeros((num_worlds,), dtype=torch.int32, device=device)

    @property
    def num_states(self) -> int:
        return self.transitions.shape[0]

    @property
    def num_actions(self) -> int:
        return self.transitions.shape[1]

    def apply_force_reset(self):
        """Resets the worlds that have force_reset set"""
        self.observations[self.force_reset] = 0
        self.force_reset[...] = 0

    def step(self):
        # Apply force_reset where needed
        self.apply_force_reset()

        # Assume self.actions has been set, index into transition matrix to get next state and reward
        # print("Actions", self.actions)
        # print("Observations", self.observations)
        # print(self.transition_rewards[self.observations])
        # print(self.transition_rewards[self.observations, self.actions])
        self.rewards[...] = self.transition_rewards[
            self.observations, self.actions
        ].squeeze()
        self.observations[...] = self.transitions[
            self.observations, self.actions
        ].squeeze()
        # self.dones[...] = (
        #     (self.observations == self.transitions.shape[0] - 1).int().squeeze()
        # )
        self.dones = self.rewards > 0
        # Reset all the dones
        # print(self.observations.shape)
        # print(self.dones.shape)
        self.observations[self.dones == 1] = 0

    def reset(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset the environment for the worlds specified by the mask.
        if not specified, reset all worlds.

        Args:
            mask: The mask of the worlds to reset. If None, resets all worlds.

        Returns:
            observations: The initial observations.
        """
        if mask is None:
            mask = torch.ones((self.num_worlds,), dtype=torch.int32, device=self.device)
        self.observations[mask] = 0
        self.force_reset[...] = 1
        self.dones[...] = 0
        self.rewards[...] = 0
        return self.observations


class TabularWorldWithStateMapper(TabularWorld):
    def __init__(self, filename: str, num_worlds: int, device: str):
        super().__init__(filename, num_worlds, device)
        initial_game_state: GameState = GameState(x=0, y=0, dir=0, object=0)
        self.game_state_tracker = GameStateTracker(
            num_worlds=num_worlds, device=device, initial_game_state=initial_game_state
        )
        self.state_mapper = TabularStateToGameStateMapper()

    def step(self):
        self.previous_observations = self.observations.clone()
        res = super().step()
        self.game_state_tracker.step(
            self.actions, self.previous_observations, self.observations
        )
        self.state_mapper.add_to_mapping(self.observations, self.game_state_tracker)
        return res

    def apply_force_reset(self):
        self.game_state_tracker.reset(mask=self.force_reset)
        return super().apply_force_reset()

    def figure_out_initial_game_state(self) -> GameState:
        """Tries to understand what is the initial game state with the least amount of information.
        This still assumes that the agent is headed to the right and not holding an object.

        Returns:
            initial_game_state: The initial game state.
        """
        return NotImplementedError
