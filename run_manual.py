import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from toolbox.printing import debug

mdp_name = "MiniGrid-DoorKey-8x8-v0"  # Replace with the name of the BRIDGE MDP.
tabular_state = 0  # Replace with the index of the tabular state.
mdp_fname = f"data_new/{mdp_name}/consolidated.npz"
mdp = np.load(mdp_fname)
transitions = mdp["transitions"]

# Get the rendered image for the tabular state.
# plt.imshow(
#     mdp["screens"][mdp["screen_mapping"][tabular_state]]
# )  # .reshape(3, 256, 256).transpose((2, 1, 0)))
# plt.show()

# Set the environment to the tabular state.
env = gym.make(f"{mdp_name}")
# help(env)
env.reset()
state_bytes = mdp["states"][
    tabular_state, : mdp["state_lengths"][tabular_state]
].tobytes()
# env.set_state(state_bytes)
env.unwrapped.render_mode = "rgb_array"

# Interactivly play
while True:
    plt.imshow(env.render())
    plt.savefig(f"game.png")
    plt.close()
    plt.imshow(
        mdp["screens"][mdp["screen_mapping"][tabular_state]]
    )  # .reshape(3, 256, 256).transpose((2, 1, 0)))
    plt.savefig(f"tabular.png")
    plt.close()
    action = int(input("action: "))
    if action == 9:
        break
    obs, reward, done, info, _ = env.step(action)
    tabular_state = transitions[tabular_state, action]
    print(f"reward: {reward}, done: {done}, info: {info}")
    if done:
        break

print("done")
# state_bytes = mdp["states"][
#     tabular_state, : mdp["state_lengths"][tabular_state]
# ].tobytes()
# env.set_state(state_bytes)
# plt.imshow(env.render())
# plt.show()
