# Filter warnings
import warnings

warnings.filterwarnings("ignore")

from typing import List

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from envs.mdp_utils import load_mdp_from_npz, get_sparse_mdp
from algorithms.value_iteration import run_value_iteration
from toolbox.printing import debug
from effective_horizon.envs.deterministic_registration import register_atari_envs
from tqdm import tqdm

register_atari_envs()

mdp_name = "atlantis_30_fs30"  # Replace with the name of the BRIDGE MDP.v

transitions, rewards = load_mdp_from_npz(
    f"/home/josselin/Downloads/bridge_dataset/mdps/{mdp_name}/consolidated.npz"
)
num_states, num_actions = transitions.shape
env = gym.make(f"BRIDGE/{mdp_name}")
obs, infos = env.reset()

frame = env.render(mode="rgb_array")
render_shape = frame.shape
render_data_shape = (num_states,) + render_shape
render_data_size = np.prod(render_data_shape) * 4 / 1024**3
obs_data_shape = (num_states,) + obs.shape
obs_data_size = np.prod(obs_data_shape) * 4 / 1024**3
print(f"Estimated render data size: {render_data_size:.2f} GB")
print(f"Estimated obs data size: {obs_data_size:.2f} GB")


visited_states = set()
path_to_state = {
    0: [],
}  # Maps a state to the sequence of actions needed from the initial state to reach it.
states_to_find = set()
states_to_visit = [0]

# First find a path to each state

with tqdm(total=num_states, desc="Finding paths", unit="states") as pbar:
    while states_to_visit:
        state: int = states_to_visit.pop(0)
        if state in visited_states:
            continue
        path: List[int] = path_to_state[state]
        neighbors = transitions[state, :]
        for action, neighbor in enumerate(neighbors):
            if neighbor not in visited_states:
                states_to_visit.append(neighbor)
                path_to_state[neighbor] = path + [action]
        visited_states.add(state)
        pbar.update(1)
    pbar.close()

# Then go through the path and render the frames
states_to_save = set(range(num_states)[::-1])
render_data = np.zeros(render_data_shape, dtype=np.uint8)
obs_data = np.zeros(obs_data_shape, dtype=np.float32)

with tqdm(
    total=num_states, desc="Rendering frames", unit="states", smoothing=0.1
) as pbar:
    while states_to_save:
        obs, _ = env.reset()
        tabular_state: int = 0  # First state
        tabular_goal_state: int = states_to_save.pop()
        path: List[int] = path_to_state[tabular_goal_state]
        for action in path:
            if (
                tabular_state in states_to_save
            ):  # We found on the way a state we want to save
                render_data[tabular_state, :] = env.render(mode="rgb_array")
                obs_data[tabular_state, :] = obs
                states_to_save.remove(tabular_state)
                pbar.update(1)
            obs, _, _, _, _ = env.step(action)
            tabular_state = transitions[tabular_state, action]

        if tabular_goal_state in states_to_save:
            render_data[tabular_goal_state, :] = env.render(mode="rgb_array")
            obs_data[tabular_goal_state, :] = obs
            states_to_save.remove(tabular_goal_state)
            pbar.update(1)
    pbar.close()
