# Code a tabular Q-learning algorithm that learns the optimal policy for the tabular world.

from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple
from toolbox.printing import debug

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
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

            # Q-learning (fast logging preparation)
            self.first_log_q_values: bool = True
            self.device = env.device
            self.coordinates = torch.tensor(
                [
                    (self.gt.mapping[tabular_state].x, self.gt.mapping[tabular_state].y)
                    for tabular_state in self.gt.mapping
                ],
                device=env.device,
            )

    def log(self, step, **kwargs):
        if not self.started:
            raise Exception("Logger not started!")

        time_s = time.time()
        if self.last_log_time_s >= 0:
            dt = time_s - self.last_log_time_s
            steps_per_s = (step - self.last_log_step) * self.num_worlds / dt
            self.add_to_buffer(steps_per_s=steps_per_s)
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

    def fast_construct_value_grid_from_tabular(self, V: torch.Tensor) -> np.ndarray:
        """Construct a grid of the V values from the tabular values.
        This function is faster than construct_value_grid_from_tabular
        because it precomputes the coordinates of each tabular state.
        in start().
        """
        grid = torch.zeros((self.gt.width, self.gt.height), device=self.device)
        count = torch.zeros((self.gt.width, self.gt.height), device=self.device)
        grid[self.coordinates[:, 0], self.coordinates[:, 1]] += V
        count[self.coordinates[:, 0], self.coordinates[:, 1]] += 1
        count[count == 0] = 1
        grid /= count
        return grid.cpu().numpy()

    def log_q_values(self, step):
        if self.log_mode != "wandb":
            return

        # Log images
        gt: GroundTruth = load_ground_truth(env_name)
        V: torch.Tensor = torch.max(self.q_values, axis=1).values
        V_grid, _ = construct_value_grid_from_tabular(
            V, gt.mapping, gt.width, gt.height, progress=False
        )
        # Does not seem to work
        # V_grid = self.fast_construct_value_grid_from_tabular(V)
        wandb.log({"Learned V": wandb.Image(V_grid.T)}, step=step * self.num_worlds)

        # Renormalize for comparison
        if np.max(V_grid) > 0:
            V_grid *= np.max(gt.V_grid) / np.max(V_grid)
        wandb.log(
            {"Difference": wandb.Image(np.abs(V_grid.T - gt.V_grid.T))},
            step=step * self.num_worlds,
        )

        if self.first_log_q_values:
            wandb.log(
                {"Ground truth V": wandb.Image(gt.V_grid.T)},
                step=step * self.num_worlds,
            )
            self.first_log_q_values = False


@dataclass
class PPOParameters:
    num_states: int  # Number of states in the environment
    num_actions: int  # Number of actions in the environment
    num_worlds: int  # Number of parallel environments
    learning_rate_start: float  # Learning rate (alpha)
    learning_rate_end: float  # Learning rate (alpha)
    discount_factor: float  # Discount factor (gamma)
    exploration_prob_start: float  # Epsilon for epsilon-greedy policy
    exploration_prob_end: float  # Epsilon for epsilon-greedy policy
    exploration_step_end: int  # Step at which epsilon reaches exploration_prob_end
    total_num_steps: int  # Maximum number of episodes for training
    max_steps_per_episode: int  # Maximum steps per episode


class PPOResults:
    def __init__(self):
        self.total_rewards = []  # List of total rewards at the end of each episode
        self.total_steps = []  # List of total steps taken in each episode
        self.q_values = []  # List of Q-values (optional, if you want to store them)


def flatten_and_apply_mask(input_tensor, mask):
    """Flatten a tensor and apply a mask.

    Args:
        input_tensor (torch.Tensor): Tensor to flatten. Shape (num_worlds, num_steps)
        mask (torch.Tensor): Mask to apply. Shape (num_worlds, num_steps)

    return non masked values
    """
    return input_tensor[mask].flatten()


class PPO:
    LOG_VALUES_EVERY = 10
    LOG_Q_VALUES_EVERY = 500000

    def __init__(
        self,
        params: PPOParameters,
        env: TabularWorld,
        log_mode: Optional[str] = None,  # "wandb",
    ):
        self.params = params
        self.env = env
        self.logger = Logger(log_mode=log_mode)

        # Initialize Q-values for each state-action pair
        self.q_values = torch.ones(
            (params.num_states, params.num_actions),
            device=env.device,
        )
        self.q_values2 = torch.ones(
            (params.num_states, params.num_actions),
            device=env.device,
        )
        # Initialize Q-values for each state-action pair
        # with Xavier initialization
        # torch.nn.init.xavier_uniform_(self.q_values)
        self.q_values = torch.nn.Parameter(self.q_values, requires_grad=True)
        self.q_values2 = torch.nn.Parameter(self.q_values2, requires_grad=True)

        # Initialize the PPO learning
        self.actor_optim = optim.Adam([self.q_values], lr=params.learning_rate_start)
        self.critic_optim = optim.Adam([self.q_values2], lr=params.learning_rate_start)

        # Initialize PPOResults
        self.results = PPOResults()

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate the value of a state.

        Args:
            state (torch.Tensor): State of the environment. Shape: (num_worlds,)

        Returns:
            torch.Tensor: Estimated value of the state. Shape: (num_worlds, num_actions)
        """
        return self.q_values2[state]

    def actor(self, state: torch.Tensor) -> torch.Tensor:
        """Return the mean of each action for a multinomial distribution.

        Args:
            state (torch.Tensor): State of the environment. Shape: (num_worlds,)

        Returns:
            torch.Tensor: Mean of each action. Shape: (num_worlds, num_actions)
        """
        Q_values = self.q_values[state]
        action_probs = torch.softmax(Q_values, dim=1)
        return action_probs

    def get_action(
        self, current_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Choose an action using a multinomial distribution over the Q-values.

        Args:
            current_state (torch.Tensor): Current state of the environment.

        Returns:
            torch.Tensor: Action to take.
            torch.Tensor: Probability of taking the action.
        """
        # epsilon = 0.5
        # if torch.rand(1) > epsilon:
        #     # Random action
        #     return torch.randint(
        #         self.params.num_actions, (self.params.num_worlds,)
        #     ).type(torch.int32).to(self.env.device), torch.zeros(
        #         (self.params.num_worlds,)
        #     ).to(
        #         self.env.device
        #     )
        # current_state is of shape (num_worlds,)
        mean = self.actor(current_state)  # (num_worlds, num_actions)
        action_probs = torch.softmax(mean, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        # Convert to int32
        action = action.type(torch.int32)
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def rollout(self, num_timesteps_needed: int):
        # Batch data. For more details, check function header.
        max_size: int = num_timesteps_needed + self.params.max_steps_per_episode - 1
        batch_obs = torch.zeros(
            (max_size,), device=self.env.device, dtype=self.env.observations.dtype
        )
        batch_acts = torch.zeros(
            (max_size,), device=self.env.device, dtype=self.env.actions.dtype
        )
        batch_log_probs = torch.zeros((max_size,), device=self.env.device)
        batch_rtgs = torch.zeros((max_size,), device=self.env.device)
        batch_lens = []

        episode_rewards: torch.Tensor = torch.zeros(
            (self.params.num_worlds, self.params.max_steps_per_episode),
            dtype=self.env.rewards.dtype,
            device=self.env.device,
        )
        episode_actions: torch.Tensor = torch.zeros(
            (self.params.num_worlds, self.params.max_steps_per_episode),
            dtype=self.env.actions.dtype,
            device=self.env.device,
        )
        episode_states: torch.Tensor = torch.zeros(
            (self.params.num_worlds, self.params.max_steps_per_episode),
            dtype=self.env.observations.dtype,
            device=self.env.device,
        )
        episode_log_probs: torch.Tensor = torch.zeros(
            (self.params.num_worlds, self.params.max_steps_per_episode),
            dtype=torch.float32,
            device=self.env.device,
        )
        episode_step_index: torch.Tensor = torch.zeros(
            (self.params.num_worlds,),
            dtype=torch.int32,
            device=self.env.device,
        )

        self.env.reset()
        t = 0
        while t < num_timesteps_needed:
            # Perform one step of the environment
            current_state: torch.Tensor = self.env.observations.clone()
            action, log_prob = self.get_action(current_state=current_state)
            self.env.actions = action
            self.env.step()

            # Store step results in buffer
            episode_rewards[:, episode_step_index] = self.env.rewards
            episode_actions[:, episode_step_index] = action
            episode_states[:, episode_step_index] = current_state
            episode_log_probs[:, episode_step_index] = log_prob
            episode_step_index += 1

            # Set environment that have exceeded the maximum number of steps to done
            if torch.any(episode_step_index >= self.params.max_steps_per_episode):
                mask_done = episode_step_index >= self.params.max_steps_per_episode
                self.env.dones[mask_done] = 1

            # Check if some trajectories finished
            if torch.any(self.env.dones):
                mask_done = self.env.dones == 1
                num_dones: int = torch.sum(mask_done).cpu().item()
                horizon: int = torch.max(episode_step_index[mask_done]).cpu().item()

                if torch.any(
                    episode_step_index[mask_done] < self.params.max_steps_per_episode
                ):
                    pass  # print("Not max steps but done!")

                # Calculate the discounted reward-to-go of each trajectory
                rtgs = torch.zeros(
                    (num_dones, horizon),
                    dtype=torch.float32,
                    device=self.env.device,
                )
                discounted_reward = torch.zeros((num_dones,), device=self.env.device)
                rewards_done = episode_rewards[mask_done]
                for t in reversed(range(horizon)):
                    discounted_reward = (
                        rewards_done[:, t]
                        + self.params.discount_factor * discounted_reward
                    )
                    rtgs[:, t] = discounted_reward

                # Flatten the data
                # Creates a mask of shape (num_dones, horizon) that is True for
                # mask[i, j] = True if the jth timestep of the ith trajectory is not done
                mask_horizon = torch.arange(horizon, device=self.env.device) < (
                    episode_step_index[mask_done].unsqueeze(1)
                )
                rtgs = flatten_and_apply_mask(rtgs, mask_horizon)
                obs = flatten_and_apply_mask(
                    episode_states[mask_done, :horizon], mask_horizon
                )
                acts = flatten_and_apply_mask(
                    episode_actions[mask_done, :horizon], mask_horizon
                )
                log_probs = flatten_and_apply_mask(
                    episode_log_probs[mask_done, :horizon], mask_horizon
                )
                num_timesteps_done = rtgs.shape[0]
                assert num_timesteps_done == obs.shape[0]
                assert num_timesteps_done == acts.shape[0]
                assert num_timesteps_done == log_probs.shape[0]
                assert num_timesteps_done == torch.sum(episode_step_index[mask_done])

                # Add the data to the batch
                end_slice = min(t + num_timesteps_done, max_size)
                batch_acts[t:end_slice] = acts[: end_slice - t]
                batch_obs[t:end_slice] = obs[: end_slice - t]
                batch_log_probs[t:end_slice] = log_probs[: end_slice - t]
                batch_rtgs[t:end_slice] = rtgs[: end_slice - t]
                batch_lens.extend(episode_step_index[mask_done].cpu().tolist())
                t += end_slice - t

                # Reset environments that are done
                self.env.reset(mask_done)
                episode_actions[mask_done] = 0
                episode_states[mask_done] = 0
                episode_log_probs[mask_done] = 0
                episode_step_index[mask_done] = 0

        # Trim batch to only filled timesteps
        batch_obs = batch_obs[:t]
        batch_acts = batch_acts[:t]
        batch_log_probs = batch_log_probs[:t]
        batch_rtgs = batch_rtgs[:t]
        # debug(batch_lens)
        batch_lens = torch.Tensor(batch_lens)
        batch_lens = batch_lens.to(self.env.device)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate(
        self, batch_obs: torch.Tensor, batch_acts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch,)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch,)

        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # debug(batch_obs)
        V = self.critic(
            batch_obs
        )  # Shape: (number of timesteps in batch, number of actions)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = torch.distributions.Categorical(mean)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def train(self):
        run_name: str = f"env={self.env.name},lr={self.params.learning_rate_start},gamma={self.params.discount_factor},eps={self.params.exploration_prob_start}-{self.params.exploration_prob_end},episodes={self.params.total_num_steps}"
        group_name: str = "ppo"
        self.logger.start(
            run_name=run_name,
            env=self.env,
            group_name=group_name,
            params=self.params,
            q_values=self.q_values,
        )

        t = 0
        i = 0
        while t < self.params.total_num_steps:
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rtgs,
                batch_lens,
            ) = self.rollout(
                2048 * 75 * 1
            )  # TODO
            batch_size = batch_obs.shape[0]
            assert batch_size <= torch.sum(batch_lens)
            t += batch_size

            # Compute the advantage
            V, _ = self.evaluate(batch_obs, batch_acts)
            # debug(V)
            # debug(batch_rtgs)
            # batch_rtgs = batch_rtgs.unsqueeze(1)
            advantage = batch_rtgs.unsqueeze(1) - V.detach()

            # Normalize the advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # This is the loop where we update our network for some n epochs
            self.clip = 0.2
            for _ in range(3):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # debug(ratios)

                # Calculate surrogate losses.
                surr1 = ratios.unsqueeze(1) * advantage
                surr2 = (
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip).unsqueeze(1)
                    * advantage
                )

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs.unsqueeze(1))
                if (i // 10) % 2 == 0:
                    loss = actor_loss
                else:
                    loss = critic_loss
                # loss = actor_loss + 0.5 * critic_loss

                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                print(
                    "Actor loss:",
                    actor_loss.detach().cpu().item(),
                    "   / Critic loss:",
                    critic_loss.detach().cpu().item(),
                )

                # Log actor loss
                # self.logger["actor_losses"].append(actor_loss.detach())
            i += 1

        # # Metrics
        # num_successes: int = (
        #     0  # Number of successes (reset because done and not max steps)
        # )
        # num_failures: int = (
        #     0  # Number of failures (reset because max steps and not done)
        # )

        # num_steps_in_episode: torch.Tensor = torch.zeros(
        #     (self.params.num_worlds), device=self.env.device
        # )

        # traj_states = torch.zeros((self.params.num_worlds,))
        # for step in tqdm(
        #     range(self.params.total_num_steps),
        #     desc="Training",
        # ):
        #     current_state: torch.Tensor = self.env.observations.clone()
        #     self.env.actions = self.choose_action(current_state=current_state)

        #     # Take actions in parallel environments
        #     self.env.step()
        #     next_state: torch.Tensor = self.env.observations

        #     # Update Q-values using the Q-learning update rule
        #     v_value_of_next = torch.max(self.q_values[next_state], axis=1).values
        #     q_target = self.env.rewards + self.params.discount_factor * v_value_of_next
        #     q_target[self.env.dones == 1] = self.env.rewards[self.env.dones == 1]
        #     self.q_values[current_state, self.env.actions] = (
        #         1 - learning_rate
        #     ) * self.q_values[
        #         current_state, self.env.actions
        #     ] + learning_rate * q_target

        #     # Update the cumulative rewards
        #     # episode_reward += self.env.rewards
        #     self.logger.add_to_buffer(avg_reward=self.env.rewards.mean().cpu().item())

        #     # Update the number of steps taken in each episode
        #     num_steps_in_episode += 1

        #     # Check if any of the environments are done
        #     if torch.any(self.env.dones):
        #         if torch.any(self.env.rewards[self.env.dones == 1] == 0):
        #             print("Reward is 0!")
        #         self.logger.add_to_buffer(
        #             avg_duration_to_success=num_steps_in_episode[self.env.dones == 1]
        #         )
        #         num_successes += torch.sum(self.env.dones == 1).cpu().item()
        #         num_steps_in_episode[self.env.dones == 1] = 0
        #         self.env.reset(self.env.dones == 1)

        #     # If some episodes are too long, reset them
        #     if torch.any(num_steps_in_episode >= self.params.max_steps_per_episode):
        #         num_failures += torch.sum(
        #             num_steps_in_episode >= self.params.max_steps_per_episode
        #         )
        #         num_steps_in_episode[
        #             num_steps_in_episode >= self.params.max_steps_per_episode
        #         ] = 0
        #         self.env.reset(
        #             num_steps_in_episode >= self.params.max_steps_per_episode
        #         )

        #     if step % self.LOG_VALUES_EVERY == 0:
        #         self.logger.log(
        #             step,
        #             epsilon=epsilon,
        #             learning_rate=learning_rate,
        #             num_successes=num_successes,
        #             num_failures=num_failures,
        #         )
        #     # if step % self.LOG_Q_VALUES_EVERY == 0:
        #     #     self.logger.log_q_values(step)

        #     # self.results.total_rewards.append(episode_reward)
        #     # self.results.total_steps.append(step + 1)
        # self.results.q_values.append(self.q_values.cpu().numpy())

        # return self.results


if __name__ == "__main__":
    # Create the tabular world
    # env_name = "MiniGrid-BlockedUnlockPickup-v0"
    env_name = "MiniGrid-Empty-8x8-v0"
    data_dir = "data/"
    file_name = f"{data_dir}/{env_name}/consolidated.npz"
    world = TabularWorld(file_name, num_worlds=4096, device="cuda")

    # Define the parameters
    params = PPOParameters(
        num_states=world.num_states,
        num_actions=world.num_actions,
        num_worlds=world.num_worlds,
        learning_rate_start=0.01,
        learning_rate_end=0.0001,
        discount_factor=0.95,
        exploration_prob_start=0.5,
        exploration_prob_end=0.1,
        exploration_step_end=2500,
        total_num_steps=4096 * 75 * 128,
        max_steps_per_episode=75,
    )

    # Create the Q-learning algorithm
    ppo = PPO(params, world)

    # Train the Q-learning algorithm
    results = ppo.train()

    # Print the results
    # print("Total rewards:", results.total_rewards)
    # print("Total steps:", results.total_steps)
    # print("Q-values:", results.q_values)

    gt: GroundTruth = load_ground_truth(env_name)
    Q: np.ndarray = ppo.q_values2.detach().cpu().numpy()
    V: np.ndarray = np.max(Q, axis=1)
    V_grid, _ = construct_value_grid_from_tabular(V, gt.mapping, gt.width, gt.height)
    rewards_grid, _ = construct_value_grid_from_tabular(
        gt.rewards, gt.mapping, gt.width, gt.height
    )
    # V_grid *= np.max(gt.V_grid) / np.max(V_grid)

    # Plot the results
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(V_grid.T)
    plt.colorbar()
    plt.title("Q-learning")
    plt.subplot(1, 4, 2)
    plt.imshow(gt.V_grid.T)
    plt.colorbar()
    plt.title("Ground truth")
    plt.subplot(1, 4, 3)
    plt.imshow(np.abs(V_grid.T - gt.V_grid.T))
    plt.colorbar()
    plt.title("Absolute difference")
    plt.subplot(1, 4, 4)
    plt.imshow(np.max(rewards_grid, axis=2).T)
    plt.colorbar()
    plt.title("Rewards")
    plt.show()
