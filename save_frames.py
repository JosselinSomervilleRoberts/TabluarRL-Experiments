# Filter warnings
import warnings

warnings.filterwarnings("ignore")

from typing import List, Dict

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from envs.mdp_utils import load_mdp_from_npz, get_sparse_mdp
from algorithms.value_iteration import run_value_iteration
from toolbox.printing import debug
from effective_horizon.envs.deterministic_registration import register_atari_envs
from tqdm import tqdm
import random

register_atari_envs()

mdp_name = "atlantis_50_fs30"  # Replace with the name of the BRIDGE MDP.v

transitions, rewards = load_mdp_from_npz(f"data_atari/{mdp_name}/consolidated.npz")
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


# First find a path to each state

num_hubs: int = num_states
hub_states: List[int] = [0]
hub_states += random.sample(range(1, num_states), num_hubs - 1)
hub_states.sort()
# print("Hub states:", hub_states)
paths_to_states_from_hubs: List[Dict[int, List[int]]] = []

if (
    True
):  # with tqdm(total=num_states * num_hubs, desc="Finding paths", unit="states") as pbar:
    for hub_state in tqdm(hub_states, desc="Hub states", unit="states"):
        visited_states = set()
        path_to_state = {
            hub_state: [],
        }  # Maps a state to the sequence of actions needed from the initial state to reach it.
        states_to_find = set()
        states_to_visit = [hub_state]

        # Path find
        while states_to_visit:
            state: int = states_to_visit.pop(0)
            if state in visited_states:
                # pbar.update(1)
                continue
            path: List[int] = path_to_state[state]
            neighbors = transitions[state, :]
            for action, neighbor in enumerate(neighbors):
                if neighbor not in visited_states:
                    states_to_visit.append(neighbor)
                    path_to_state[neighbor] = path + [action]
            visited_states.add(state)
            # pbar.update(1)

        # Save the path
        paths_to_states_from_hubs.append(path_to_state)
    # pbar.close()

distances = np.full((num_states, num_states), np.inf)

# Init distance from start to end as 1 + distance from 0 to end
for state in range(num_states):
    path_from_0_to_state = paths_to_states_from_hubs[0][state]
    distance_from_0_to_state = len(path_from_0_to_state)
    distances[:, state] = distance_from_0_to_state + 1

for start in range(num_states):
    paths = paths_to_states_from_hubs[start]
    for end, path in paths.items():
        distances[start, end] = len(path)

# Set distances to itself to inf
for i in range(num_states):
    distances[i, i] = np.inf

# # Build 2 N x N matrix representing the shortest path from any state to any other state
# # - dist[i,j] is the distance from state i to state j
# # - pred[i,j] is the action to take from state i to reach state j
# #   We add action num_actions to reset the environment (i.e. go back to the initial state)
# distances = np.full((num_states, num_states), np.inf)
# pred = np.full((num_states, num_states), -1, dtype=int)

# # Setting distance to self as zero
# for i in range(num_states):
#     distances[i, i] = 0

# # Set direct transitions
# for s in range(num_states):
#     for a in range(num_actions):
#         next_state = transitions[s, a]
#         if next_state != s:
#             distances[s, next_state] = 1  # One step to transition
#             pred[s, next_state] = a  # Action taken

# # For all states set distance to 1 + distance from 0 to destination
# # Set action as reset (num_actions)
# for state in range(num_states):
#     path_from_0_to_state = paths_to_states_from_hubs[0][state]
#     distance_from_0_to_state = len(path_from_0_to_state)
#     distances[:, state] = distance_from_0_to_state + 1
#     pred[:, state] = num_actions

# for state in range(num_states):
#     path = paths_to_states_from_hubs[0][state]
#     for action in path:
#         next_state = transitions[state, action]
#         distances[state, next_state] = 1
#         pred[state, next_state] = action


# Then go through the path and render the frames
states_to_save = set(range(num_states)[::-1])
render_data = np.zeros(render_data_shape, dtype=np.uint8)
print("Render data shape:", render_data.shape)
print("0 in states to save:", 0 in states_to_save)
obs_data = np.zeros(obs_data_shape, dtype=np.float32)

# with tqdm(
#     total=num_states, desc="Rendering frames", unit="states", smoothing=0.1
# ) as pbar:
#     i = 0
#     while states_to_save and i < 1000000:
#         obs, _ = env.reset()
#         tabular_state: int = 0  # First state
#         tabular_goal_state: int = states_to_save.pop()
#         states_to_save.add(tabular_goal_state)
#         path: List[int] = path_to_state[tabular_goal_state]
#         for action in path:
#             if (
#                 tabular_state in states_to_save
#             ):  # We found on the way a state we want to save
#                 rendered: np.ndarray = env.render(mode="rgb_array")
#                 render_data[tabular_state] = rendered.astype(np.uint8)
#                 obs_data[tabular_state] = obs
#                 states_to_save.remove(tabular_state)
#                 pbar.update(1)
#             obs, _, _, _, _ = env.step(action)
#             tabular_state = transitions[tabular_state, action]
#             i += 1

#         if tabular_goal_state in states_to_save:
#             render_data[tabular_goal_state] = env.render(mode="rgb_array")
#             obs_data[tabular_goal_state] = obs
#             states_to_save.remove(tabular_goal_state)
#             pbar.update(1)
#     pbar.close()

# # Floyd version that does not work

# # Filter warnings
# import warnings

# warnings.filterwarnings("ignore")

# from typing import List, Tuple

# import numpy as np
# import gymnasium as gym
# import matplotlib.pyplot as plt
# from envs.mdp_utils import load_mdp_from_npz, get_sparse_mdp
# from algorithms.value_iteration import run_value_iteration
# from toolbox.printing import debug
# from effective_horizon.envs.deterministic_registration import register_atari_envs
# from tqdm import tqdm

# register_atari_envs()

# mdp_name = "atlantis_30_fs30"  # Replace with the name of the BRIDGE MDP.v

# transitions, rewards = load_mdp_from_npz(
#     f"/home/josselin/Downloads/bridge_dataset/mdps/{mdp_name}/consolidated.npz"
# )
# num_states, num_actions = transitions.shape
# env = gym.make(f"BRIDGE/{mdp_name}")
# obs, infos = env.reset()

# frame = env.render(mode="rgb_array")
# render_shape = frame.shape
# render_data_shape = (num_states,) + render_shape
# render_data_size = np.prod(render_data_shape) * 4 / 1024**3
# obs_data_shape = (num_states,) + obs.shape
# obs_data_size = np.prod(obs_data_shape) * 4 / 1024**3
# print(f"Estimated render data size: {render_data_size:.2f} GB")
# print(f"Estimated obs data size: {obs_data_size:.2f} GB")
# print("Number of states:", num_states)
# print("Number of actions:", num_actions)
# print("Last transition:", transitions[num_states - 1, :])


# def floyd_warshall(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     N, A = T.shape
#     # Initialize distance matrix with infinity, and predecessor matrix with None
#     dist = np.full((N, N), np.inf)
#     pred = np.full((N, N), -1, dtype=int)

#     # Setting distance to self as zero
#     for i in range(N):
#         dist[i, i] = 0

#     # Fill initial values based on direct transitions
#     new_trans = set()
#     for s in range(N):
#         for a in range(A):
#             next_state = T[s, a]
#             if next_state != s:
#                 dist[s, next_state] = 1  # One step to transition
#                 pred[s, next_state] = a  # Action taken
#                 new_trans.add((s, next_state))

#     # Refinement state.
#     # The algorithm goes as follows:
#     # For each pair (s,sp) in new_trans, set the neighbors of sp to True in the table update.
#     # For each pair (s,spp) in update set the distance to i+1 if dist[s,spp] > i+1.
#     # Update the predecessor matrix accordingly.
#     # Repeat until no more updates are made.
#     # For efficiency, instead of storing matrics, store list of indices
#     horizon = 100
#     dist_init = 2
#     pred_buffer_1 = np.full((N, N), -1, dtype=int)
#     pred_buffer_2 = np.full((N, N), -1, dtype=int)
#     for i in tqdm(range(horizon)):
#         update = set()

#         # Update the update table
#         for s, sp in new_trans:
#             for a in range(A):
#                 spp = T[sp, a]
#                 if spp != sp and spp != s:
#                     update.add((s, spp))
#                     pred_buffer_1[s, spp] = a
#                     pred_buffer_2[s, spp] = sp

#         # Empty new_trans
#         new_trans = set()

#         # Update the distance matrix
#         cur_dist = i + dist_init
#         for s, spp in update:
#             if dist[s, spp] > cur_dist:
#                 sp = pred_buffer_2[s, spp]
#                 a = pred_buffer_1[s, spp]
#                 dist[s, spp] = cur_dist
#                 pred[sp, spp] = a
#                 new_trans.add((s, spp))

#         # if not has_been_updated:
#         #     break

#     # Print the maximum distance
#     # for i in range(100):
#     #     print(f"Num elts different than inf: {np.sum(dist[i] < np.inf)}")

#     return dist, pred

visited_states = set()
path_to_state = {
    0: [],
}  # Maps a state to the sequence of actions needed from the initial state to reach it.
states_to_find = set()
states_to_visit = [0]

# Then go through the path and render the frames
states_to_save = set(range(num_states))
render_data = np.zeros(render_data_shape, dtype=np.uint8)
obs_data = np.zeros(obs_data_shape, dtype=np.float32)

with tqdm(
    total=num_states, desc="Rendering frames", unit="states", smoothing=0.1
) as pbar:
    obs, _ = env.reset()
    tabular_state: int = 0  # First state

    # Save the first state
    render_data[tabular_state] = env.render(mode="rgb_array")
    obs_data[tabular_state] = obs
    states_to_save.remove(tabular_state)
    distances[:, tabular_state] = np.inf
    pbar.update(1)

    while states_to_save:
        # Choose a goal state
        # Greedy: we choose the closest one
        local_dist = distances[tabular_state, :]
        tabular_goal_state: int = np.argmin(local_dist)
        # print("Goal state:", tabular_goal_state)

        if tabular_goal_state not in paths_to_states_from_hubs[tabular_state]:
            # No direct path to the goal state, we need to reset
            # print("No direct path to the goal state, we need to reset")
            tabular_state = 0
            obs, _ = env.reset()
        else:
            # Find the path to the goal state
            path: List[int] = paths_to_states_from_hubs[tabular_state][
                tabular_goal_state
            ]
            # print("Path:", path)

            for action in path:
                obs, _, _, _, _ = env.step(action)
                tabular_state = transitions[tabular_state, action]

                if tabular_state in states_to_save:
                    # We found on the way a state we want to save
                    rendered: np.ndarray = env.render(mode="rgb_array")
                    render_data[tabular_state] = rendered.astype(np.uint8)
                    obs_data[tabular_state] = obs
                    states_to_save.remove(tabular_state)
                    distances[:, tabular_state] = np.inf
                    pbar.update(1)

        # print("New state:", tabular_state)

    pbar.close()

# Save the data
np.save(f"render_data_{mdp_name}.npy", render_data)
np.save(f"obs_data_{mdp_name}.npy", obs_data)
