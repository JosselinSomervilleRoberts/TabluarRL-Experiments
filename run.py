# Filter warnings
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from envs.mdp_utils import load_mdp_from_npz, get_sparse_mdp
from algorithms.value_iteration import run_value_iteration


mdp_name = "MiniGrid-Empty-16x16-v0"  # Replace with the name of the BRIDGE MDP.v

transitions, rewards = load_mdp_from_npz(
    f"/home/josselin/Downloads/bridge_dataset/mdps/{mdp_name}/consolidated.npz"
)
num_states, num_actions = transitions.shape
done_state = num_states - 1
tabular_state = 214  # Replace with the index of the tabular state.
print("Number of states:", num_states)
print("Number of actions:", num_actions)

# Find indices of rewards equal to 1
print("Rewards equal to 1:", np.argwhere(rewards == 1))
print("Rewards n - 1:", rewards[num_states - 1, :])
# one_hot_rewards = np.zeros((num_states, num_actions))
# one_hot_rewards[tabular_state, :] = 1

sparse_transitions, rewards_vector = get_sparse_mdp(transitions, rewards)
vi = run_value_iteration(sparse_transitions, rewards_vector, horizon=100, gamma=0.995)
print("Done!")

env = gym.make(f"{mdp_name}")
obs, infos = env.reset()
print(obs)
# Save obs in a txt file
# np.savetxt("obs.txt", obs[0])

state, t = 0, 0
while state != tabular_state:
    action = np.argmax(vi.optimal_qs[0, state, :])
    obs, reward, _, _, _ = env.step(action)
    print("t", t, "Reward:", reward)
    if reward > 0:
        print("Reward reached!")
        break
    # print(obs)
    state = transitions[state, action]
    t += 1
    env.unwrapped.render_mode = "rgb_array"
    plt.imshow(env.render())
    plt.savefig(f"step_{t}.png")
    print("t =", t, "state =", state)
print("State reached!")
