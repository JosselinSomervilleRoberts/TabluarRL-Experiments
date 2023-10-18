# Filter warnings
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from envs.mapping import construct_mapping
from algorithms.value_iteration import run_value_iteration
from envs.tabular_world import TabularWorld, GameState
from envs.mdp_utils import get_sparse_mdp, load_mdp_from_npz


mdp_name = "MiniGrid-MultiRoom-N6-v0"  # Replace with the name of the BRIDGE MDP.v
file_name = f"/home/josselin/Downloads/bridge_dataset/mdps/{mdp_name}/consolidated.npz"
world = TabularWorld(file_name, num_worlds=1, device="cpu")

# Run value iteration
transitions, rewards = load_mdp_from_npz(file_name)
sparse_transitions, rewards_vector = get_sparse_mdp(transitions, rewards)
vi = run_value_iteration(sparse_transitions, rewards_vector, horizon=100, gamma=0.96)
Q = vi.optimal_qs[0]
V = vi.optimal_values[0]
print("V shape:", V.shape)
print("Q shape:", Q.shape)

# Construct mapping
mapping = construct_mapping(world)
print("Mapping constructed!")
print("Number of states in the world:", world.num_states)
print("Number of states in mapping:", len(mapping))

# Figure out xmin, ymin, xmax, ymax
xmin = min([game_state.x for game_state in mapping.values()])
xmax = max([game_state.x for game_state in mapping.values()])
ymin = min([game_state.y for game_state in mapping.values()])
ymax = max([game_state.y for game_state in mapping.values()])
width = xmax - xmin + 3
height = ymax - ymin + 3

# Make a grid of the values
grid = np.zeros((width, height))
count = np.zeros((width, height))
for tabular_state in tqdm(mapping, desc="Making grid"):
    game_state: GameState = mapping[tabular_state]
    if game_state.object != 0:
        continue
    grid[1 + game_state.x - xmin, 1 + game_state.y - ymin] += V[tabular_state]
    count[1 + game_state.x - xmin, 1 + game_state.y - ymin] += 1
grid = grid / np.maximum(count, 1)


def draw_optimal_path(
    transitions: np.ndarray,
    Q: np.array,
    xmin: int,
    ymin: int,
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
        done_tabular_state = len(V) - 1

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
    for i in range(len(path)):
        # Draw an arrow of length 0.4 from the center of the current state in the game state direction
        # The arrow should be blue if object is 0, red if object is 1
        game_state: GameState = mapping[path[i]]
        print("t =", i, "game_state =", game_state)
        color = "blue" if game_state.object == 0 else "red"
        plt.arrow(
            1.0 + game_state.x - xmin,
            1.0 + game_state.y - ymin,
            0.2 * np.cos(game_state.dir * np.pi / 2),
            0.2 * np.sin(game_state.dir * np.pi / 2),
            color=color,
            width=0.05,
        )


# Plot the grid
plt.imshow(grid.T)
draw_optimal_path(transitions, Q, xmin, ymin)
plt.colorbar()
plt.savefig("grid.png")
plt.show()
