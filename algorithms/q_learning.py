# Code a tabular Q-learning algorithm that learns the optimal policy for the tabular world.

from dataclasses import dataclass
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from tqdm import tqdm
import wandb

from algorithms.constants import WANDB_PROJECT_NAME
from envs.tabular_world import TabularWorld
from envs.ground_truth import load_ground_truth, GroundTruth
from envs.mapping import construct_value_grid_from_tabular


class LoggingBuffer:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer = np.zeros(max_size, dtype=np.float32)
        self.index = 0
        self.count = 0

    def append(self, value: Union[float, torch.Tensor]):
        if isinstance(value, torch.Tensor):
            value = value.mean().cpu().item()
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.max_size
        if self.count < self.max_size:
            self.count += 1

    def mean(self):
        return np.mean(self.buffer[: self.count])


class Logger:
    def __init__(self, log_mode: Optional[str] = "wandb"):
        self.log_mode: Optional[str] = log_mode
        self.started: bool = False
        self.buffers: Dict[str, LoggingBuffer] = {}

    def start(
        self,
        run_name: str,
        env: TabularWorld,
        group_name: Optional[str] = None,
        params: Optional[dict] = None,
        q_values: Optional[torch.Tensor] = None,
    ):
        if self.log_mode == "wandb":
            wandb.init(
                project=WANDB_PROJECT_NAME,
                group=group_name,
                name=run_name,
                monitor_gym=True,
                save_code=True,
                config=params,
            )
            self.q_values = q_values

        if self.log_mode is not None:
            self.num_worlds = env.num_worlds
            self.gt = load_ground_truth(env.name)
            self.started = True
            self.last_log_time_s = -1
            self.last_log_step = -1

    def log(self, step, **kwargs):
        if not self.started:
            raise Exception("Logger not started!")

        time_s = time.time()
        if self.last_log_time_s >= 0:
            dt = time_s - self.last_log_time_s
            steps_per_s = (step - self.last_log_step) * self.num_worlds / dt
            kwargs["steps_per_s"] = steps_per_s
        self.last_log_time_s = time_s
        self.last_log_step = step

        if self.log_mode == "wandb":
            wandb.log(kwargs, step=step * self.num_worlds)

            for title, buffer in self.buffers.items():
                wandb.log({title: buffer.mean()}, step=step * self.num_worlds)

    def add_to_buffer(self, **kwargs):
        if not self.started:
            raise Exception("Logger not started!")

        if self.log_mode == "wandb":
            for title, value in kwargs.items():
                if title not in self.buffers:
                    self.buffers[title] = LoggingBuffer()
                self.buffers[title].append(value)

    def log_q_values(self, step):
        if self.log_mode != "wandb":
            return

        # Creates a mapping of titles to images to log
        images: Dict[str, np.ndarray] = {}
        gt: GroundTruth = load_ground_truth(env_name)
        Q: np.ndarray = self.q_values.cpu().numpy()
        V: np.ndarray = np.max(Q, axis=1)
        V_grid, _ = construct_value_grid_from_tabular(
            V, gt.mapping, gt.width, gt.height
        )
        images["Learned V"] = V_grid.T.copy()

        # Renormalize for comparison
        if np.max(V_grid) > 0:
            V_grid *= np.max(gt.V_grid) / np.max(V_grid)
        images["Difference"] = np.abs(V_grid.T - gt.V_grid.T).copy()

        images["Ground truth V"] = gt.V_grid.T.copy()

        # Log the images
        if self.log_mode == "wandb":
            for title, image in images.items():
                wandb.log({title: wandb.Image(image)}, step=step * self.num_worlds)


@dataclass
class QLearningParameters:
    num_states: int  # Number of states in the environment
    num_actions: int  # Number of actions in the environment
    num_worlds: int  # Number of parallel environments
    learning_rate: float  # Learning rate (alpha)
    discount_factor: float  # Discount factor (gamma)
    exploration_prob_start: float  # Epsilon for epsilon-greedy policy
    exploration_prob_end: float  # Epsilon for epsilon-greedy policy
    total_num_steps: int  # Maximum number of episodes for training
    max_steps_per_episode: int  # Maximum steps per episode


class QLearningResults:
    def __init__(self):
        self.total_rewards = []  # List of total rewards at the end of each episode
        self.total_steps = []  # List of total steps taken in each episode
        self.q_values = []  # List of Q-values (optional, if you want to store them)


class QLearning:
    LOG_VALUES_EVERY = 10
    LOG_Q_VALUES_EVERY = 100

    def __init__(
        self,
        params: QLearningParameters,
        env: TabularWorld,
        log_mode: Optional[str] = "wandb",
    ):
        self.params = params
        self.env = env
        self.logger = Logger(log_mode=log_mode)

        # Initialize Q-values for each state-action pair
        self.q_values = torch.zeros(
            (params.num_states, params.num_actions),
            device=env.device,
        )

        # Initialize QLearningResults
        self.results = QLearningResults()

    def train(self):
        run_name: str = f"env={self.env.name},lr={self.params.learning_rate},gamma={self.params.discount_factor},eps={self.params.exploration_prob_start}-{self.params.exploration_prob_end},episodes={self.params.total_num_steps}"
        group_name: str = "q-learning"
        self.logger.start(
            run_name=run_name,
            env=self.env,
            group_name=group_name,
            params=self.params,
            q_values=self.q_values,
        )

        # Metrics
        num_successes: int = (
            0  # Number of successes (reset because done and not max steps)
        )
        num_failures: int = (
            0  # Number of failures (reset because max steps and not done)
        )

        num_steps_in_episode: torch.Tensor = torch.zeros(
            (self.params.num_worlds), device=self.env.device
        )
        for step in tqdm(
            range(self.params.total_num_steps),
            desc="Training",
        ):
            self.env.reset()  # Reset all parallel environments
            # episode_reward = torch.zeros(
            #     (self.params.num_worlds), device=self.env.device
            # )
            epsilon = (
                self.params.exploration_prob_start
                + (
                    self.params.exploration_prob_end
                    - self.params.exploration_prob_start
                )
                * step
                / self.params.total_num_steps
            )
            learning_rate = self.params.learning_rate

            current_state: torch.Tensor = self.env.observations.clone()

            # Choose actions using epsilon-greedy policy
            # Some environment will explore
            explore = (
                torch.rand((self.params.num_worlds), device=self.env.device) < epsilon
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
            q_target = self.env.rewards + self.params.discount_factor * v_value_of_next
            self.q_values[current_state, self.env.actions] = (
                1 - learning_rate
            ) * self.q_values[
                current_state, self.env.actions
            ] + learning_rate * q_target

            # Update the cumulative rewards
            # episode_reward += self.env.rewards
            self.logger.add_to_buffer(avg_reward=self.env.rewards.mean().cpu().item())

            # Update the number of steps taken in each episode
            num_steps_in_episode += 1

            # Check if any of the environments are done
            if torch.any(self.env.dones):
                self.logger.add_to_buffer(
                    avg_duration_to_success=num_steps_in_episode[self.env.dones == 1]
                )
                num_successes += torch.sum(self.env.dones == 1).cpu().item()
                num_steps_in_episode[self.env.dones == 1] = 0
                self.env.reset(self.env.dones == 1)

            # If some episodes are too long, reset them
            if torch.any(num_steps_in_episode >= self.params.max_steps_per_episode):
                num_failures += torch.sum(
                    num_steps_in_episode >= self.params.max_steps_per_episode
                )
                num_steps_in_episode[
                    num_steps_in_episode >= self.params.max_steps_per_episode
                ] = 0
                self.env.reset(
                    num_steps_in_episode >= self.params.max_steps_per_episode
                )

            if step % self.LOG_VALUES_EVERY == 0:
                self.logger.log(
                    step,
                    epsilon=epsilon,
                    learning_rate=learning_rate,
                    num_successes=num_successes,
                    num_failures=num_failures,
                )
            if step % self.LOG_Q_VALUES_EVERY == 0:
                self.logger.log_q_values(step)

            # self.results.total_rewards.append(episode_reward)
            # self.results.total_steps.append(step + 1)
        self.results.q_values.append(self.q_values.cpu().numpy())

        return self.results


if __name__ == "__main__":
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
        total_num_steps=2000,
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
