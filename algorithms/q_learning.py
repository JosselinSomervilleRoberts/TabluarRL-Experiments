# Code a tabular Q-learning algorithm that learns the optimal policy for the tabular world.

from dataclasses import dataclass

import torch
from tqdm import tqdm

from envs.tabular_world import TabularWorld, GameState


@dataclass
class QLearningParameters:
    num_states: int  # Number of states in the environment
    num_actions: int  # Number of actions in the environment
    num_worlds: int  # Number of parallel environments
    learning_rate: float  # Learning rate (alpha)
    discount_factor: float  # Discount factor (gamma)
    exploration_prob_start: float  # Epsilon for epsilon-greedy policy
    exploration_prob_end: float  # Epsilon for epsilon-greedy policy
    max_num_episodes: int  # Maximum number of episodes for training
    max_steps_per_episode: int  # Maximum steps per episode


class QLearningResults:
    def __init__(self):
        self.total_rewards = []  # List of total rewards at the end of each episode
        self.total_steps = []  # List of total steps taken in each episode
        self.q_values = []  # List of Q-values (optional, if you want to store them)


class QLearning:
    def __init__(self, params: QLearningParameters, env: TabularWorld):
        self.params = params
        self.env = env

        # Initialize Q-values for each state-action pair
        self.q_values = torch.zeros(
            (params.num_states, params.num_actions),
            device=env.device,
        )

        # Initialize QLearningResults
        self.results = QLearningResults()

    def train(self):
        for episode in tqdm(range(self.params.max_num_episodes), desc="Training"):
            self.env.reset()  # Reset all parallel environments
            episode_reward = torch.zeros(
                (self.params.num_worlds), device=self.env.device
            )
            epsilon = (
                self.params.exploration_prob_start
                + (
                    self.params.exploration_prob_end
                    - self.params.exploration_prob_start
                )
                * episode
                / self.params.max_num_episodes
            )

            for step in range(self.params.max_steps_per_episode):
                current_state: torch.Tensor = self.env.observations.clone()

                # Choose actions using epsilon-greedy policy
                # Some environment will explore
                explore = (
                    torch.rand((self.params.num_worlds), device=self.env.device)
                    < epsilon
                )
                self.env.actions[explore] = torch.randint(
                    self.params.num_actions,
                    (torch.sum(explore),),
                    dtype=self.env.actions.dtype,
                    device=self.env.device,
                )
                # Others will exploit
                exploit = ~explore
                Q_values = self.q_values[current_state[exploit]]
                self.env.actions[exploit] = torch.argmax(Q_values, axis=1).int()

                # Take actions in parallel environments
                self.env.step()
                next_state: torch.Tensor = self.env.observations

                # Update Q-values using the Q-learning update rule
                v_value_of_next = torch.max(self.q_values[next_state], axis=1).values
                q_target = (
                    self.env.rewards + self.params.discount_factor * v_value_of_next
                )
                self.q_values[current_state, self.env.actions] = (
                    1 - self.params.learning_rate
                ) * self.q_values[
                    current_state, self.env.actions
                ] + self.params.learning_rate * q_target

                # Update the cumulative rewards
                episode_reward += self.env.rewards

                # Check if any of the environments are done
                if torch.any(self.env.dones):
                    break

            # self.results.total_rewards.append(episode_reward)
            # self.results.total_steps.append(step + 1)
        self.results.q_values.append(self.q_values.cpu().numpy())

        return self.results


if __name__ == "__main__":
    from envs.ground_truth import load_ground_truth, GroundTruth
    from envs.mapping import construct_value_grid_from_tabular
    import matplotlib.pyplot as plt
    import numpy as np

    # Create the tabular world
    env_name = "MiniGrid-BlockedUnlockPickup-v0"
    # env_name = "MiniGrid-Empty-8x8-v0"
    data_dir = "data/"
    file_name = f"{data_dir}/{env_name}/consolidated.npz"
    world = TabularWorld(file_name, num_worlds=32000, device="cuda")

    # Define the parameters
    params = QLearningParameters(
        num_states=world.num_states,
        num_actions=world.num_actions,
        num_worlds=world.num_worlds,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_prob_start=1.0,
        exploration_prob_end=0.05,
        max_num_episodes=4000,
        max_steps_per_episode=100,
    )

    # Create the Q-learning algorithm
    q_learning = QLearning(params, world)

    # Train the Q-learning algorithm
    results = q_learning.train()

    # Print the results
    # print("Total rewards:", results.total_rewards)
    # print("Total steps:", results.total_steps)
    # print("Q-values:", results.q_values)

    gt: GroundTruth = load_ground_truth(env_name)
    Q: np.ndarray = results.q_values[-1]
    V: np.ndarray = np.max(Q, axis=1)
    print("Q shape:", Q.shape)
    V_grid, _ = construct_value_grid_from_tabular(V, gt.mapping, gt.width, gt.height)
    print("V grid shape:", V_grid.shape)
    V_grid *= np.max(gt.V_grid) / np.max(V_grid)

    # Plot the results
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(V_grid.T)
    plt.colorbar()
    plt.title("Q-learning")
    plt.subplot(1, 3, 2)
    plt.imshow(gt.V_grid.T)
    plt.colorbar()
    plt.title("Ground truth")
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(V_grid.T - gt.V_grid.T))
    plt.colorbar()
    plt.title("Absolute difference")
    plt.show()
