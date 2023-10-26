############################### Import libraries ###############################

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_buffer_size: int = 10000,
        device: torch.device = torch.device("cpu"),
    ):
        self.actions = torch.zeros(
            (max_buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self.states = torch.zeros(
            (max_buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self.logprobs = torch.zeros(
            (max_buffer_size, 1), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(
            (max_buffer_size, 1), dtype=torch.float32, device=device
        )
        self.state_values = torch.zeros(
            (max_buffer_size, 1), dtype=torch.float32, device=device
        )
        self.is_terminals = torch.zeros(
            (max_buffer_size, 1), dtype=torch.float32, device=device
        )
        self.num_entries: int = 0
        self.index: int = 0
        self.max_buffer_size: int = max_buffer_size

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        state_value: torch.Tensor,
        is_terminal: torch.Tensor,
    ) -> None:
        """Adds a transition to the buffer

        Args:
            state (torch.Tensor): The state to add (Shape: (state_dim,))
            action (torch.Tensor): The action to add (Shape: (action_dim,))
            logprob (torch.Tensor): The log probability of the action (Shape: (1,))
            reward (torch.Tensor): The reward for the transition (Shape: (1,))
            state_value (torch.Tensor): The value of the state (Shape: (1,))
            is_terminal (torch.Tensor): Whether the state is terminal (Shape: (1,))
        """
        self.actions[self.index] = action
        self.states[self.index] = state
        self.logprobs[self.index] = logprob
        self.rewards[self.index] = reward
        self.state_values[self.index] = state_value
        self.is_terminals[self.index] = is_terminal
        self.index = (self.index + 1) % self.max_buffer_size
        self.num_entries = min(self.num_entries + 1, self.max_buffer_size)

    def add_batch(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        state_values: torch.Tensor,
        is_terminals: torch.Tensor,
        num_entries: int,
    ):
        """Adds the contents of another buffer to this buffer

        Args:
            actions (torch.Tensor): The actions to add (Shape: (num_entries, action_dim))
            states (torch.Tensor): The states to add (Shape: (num_entries, state_dim))
            logprobs (torch.Tensor): The log probabilities of the actions (Shape: (num_entries, 1))
            rewards (torch.Tensor): The rewards for the transitions (Shape: (num_entries, 1))
            state_values (torch.Tensor): The values of the states (Shape: (num_entries, 1))
            is_terminals (torch.Tensor): Whether the states are terminal (Shape: (num_entries, 1))
        """
        # print(f"Adding {num_entries} entries to buffer")
        if num_entries > self.max_buffer_size:
            raise ValueError("Buffer too large")

        # Split into bulks so that if fits in the buffer (at most 2 bulks):
        # - from self.index to min(self.index + buffer.num_entries, self.max_buffer_size)
        # - from 0 to max(0, self.index + buffer.num_entries - self.max_buffer_size)

        # First bulk
        bulk_size = min(num_entries, self.max_buffer_size - self.index)
        self.actions[self.index : self.index + bulk_size] = actions[:bulk_size]
        self.states[self.index : self.index + bulk_size] = states[:bulk_size]
        self.logprobs[self.index : self.index + bulk_size] = logprobs[:bulk_size]
        self.rewards[self.index : self.index + bulk_size] = rewards[:bulk_size]
        self.state_values[self.index : self.index + bulk_size] = state_values[
            :bulk_size
        ]
        self.is_terminals[self.index : self.index + bulk_size] = is_terminals[
            :bulk_size
        ]
        self.num_entries = min(self.num_entries + bulk_size, self.max_buffer_size)
        self.index = (self.index + bulk_size) % self.max_buffer_size

        # Second bulk
        if bulk_size == num_entries:
            return  # Everything has already been added
        bulk_size = num_entries - bulk_size
        self.actions[:bulk_size] = actions[-bulk_size:]
        self.states[:bulk_size] = states[-bulk_size:]
        self.logprobs[:bulk_size] = logprobs[-bulk_size:]
        self.rewards[:bulk_size] = rewards[-bulk_size:]
        self.state_values[:bulk_size] = state_values[-bulk_size:]
        self.is_terminals[:bulk_size] = is_terminals[-bulk_size:]
        self.num_entries = (
            self.max_buffer_size
        )  # If we reached here, it means we looped around
        self.index = bulk_size

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Samples a batch from the buffer

        Args:
            batch_size (int): The size of the batch to sample

        Returns:
            Tuple[torch.Tensor, ...]: The batch of transitions
        """
        # Indices go from num_entries - batch_size to num_entries
        # indices = torch.randint(0, self.num_entries, (batch_size,))
        return (
            self.states[: self.num_entries].detach().squeeze(),
            self.actions[: self.num_entries].detach().squeeze(),
            self.logprobs[: self.num_entries].detach().squeeze(),
            self.rewards[: self.num_entries].detach().squeeze(),
            self.state_values[: self.num_entries].detach().squeeze(),
            self.is_terminals[: self.num_entries].detach().squeeze(),
        )

    def clear(self) -> None:
        """Clears the buffer"""
        self.num_entries = 0
        self.index = 0


class BatchedRolloutBuffer:
    """A buffer that stores episodes as batches"""

    def __init__(
        self,
        max_ep_len: int,
        state_dim: int,
        batch_size: int,
        action_dim: int,
        max_buffer_size: int = 1000000,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_ep_len = max_ep_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = device

        self.states = torch.zeros(
            (batch_size, max_ep_len, state_dim), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (batch_size, max_ep_len, action_dim), dtype=torch.int64, device=device
        )
        self.logprobs = torch.zeros(
            (batch_size, max_ep_len, 1), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(
            (batch_size, max_ep_len, 1), dtype=torch.float32, device=device
        )
        self.state_values = torch.zeros(
            (batch_size, max_ep_len, 1), dtype=torch.float32, device=device
        )
        self.is_terminals = torch.zeros(
            (batch_size, max_ep_len, 1), dtype=bool, device=device
        )
        self.episode_indices = torch.zeros(
            (batch_size,), dtype=torch.int32, device=device
        )

        self.linear_buffer = RolloutBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_buffer_size=max_buffer_size,
            device=device,
        )

    def batch_add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        state_values: torch.Tensor,
        is_terminals: torch.Tensor,
    ) -> None:
        # All tensors are of shape (batch_size, state_dim/action_dim/1)
        self.states[:, self.episode_indices] = states
        self.actions[:, self.episode_indices] = actions
        self.logprobs[:, self.episode_indices] = logprobs
        self.rewards[:, self.episode_indices] = rewards
        self.state_values[:, self.episode_indices] = state_values
        self.is_terminals[:, self.episode_indices] = is_terminals
        self.episode_indices += 1

        # Handle episodes that are done and add them to the linear buffer
        for i in range(self.batch_size):
            if is_terminals[i]:
                # print(f"Terminal {i} at {self.episode_indices[i]}")
                self.linear_buffer.add_batch(
                    actions=self.actions[i],
                    states=self.states[i],
                    logprobs=self.logprobs[i],
                    rewards=self.rewards[i],
                    state_values=self.state_values[i],
                    is_terminals=self.is_terminals[i],
                    num_entries=self.episode_indices[i].item(),
                )
                self.episode_indices[i] = 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        return self.linear_buffer.sample(batch_size)

    def clear(self) -> None:
        self.linear_buffer.clear()
        self.episode_indices[:] = 0


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: tuple,
        action_dim: tuple,
        has_continuous_action_space: bool = False,
        action_std_init: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super(Policy, self).__init__()
        self.device: torch.device = device
        self.state_dim: tuple = state_dim
        self.action_dim: tuple = action_dim
        self.has_continuous_action_space: bool = has_continuous_action_space
        self.action_std: Optional[float] = action_std_init
        if self.has_continuous_action_space and self.action_std is None:
            raise ValueError("Must set action_std_init with continuous action space")
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init
            ).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the action to take given a state

        Args:
            state (torch.Tensor): The state to calculate the action for (Shape: (batch_size, state_dim))

        Returns:
            action (torch.Tensor): The action to take (Shape: (batch_size, action_dim))
            action_logprob (torch.Tensor): The log probability of the action (Shape: (batch_size, 1))
            state_val (torch.Tensor): The value of the state (Shape: (batch_size, 1))
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the log probability of an action given a state and the value of the state

        Args:
            state (torch.Tensor): The state to calculate the action for (Shape: (batch_size, state_dim))
            action (torch.Tensor): The action to evaluate (Shape: (batch_size, action_dim))

        Returns:
            action_logprobs (torch.Tensor): The log probability of the action (Shape: (batch_size, 1))
            state_values (torch.Tensor): The value of the state (Shape: (batch_size, 1))
            dist_entropy (torch.Tensor): The entropy of the distribution (Shape: (batch_size, 1))
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var, device=self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def set_action_std(self, new_action_std: float) -> None:
        """Sets the standard deviation of the action distribution

        Args:
            new_action_std (float): The new standard deviation to use
        """
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std, device=self.device
            )
        else:
            print("-------------------------------------------------------------------")
            print("WARNING : Calling Policy::set_action_std() on discrete action space")
            print("-------------------------------------------------------------------")


class DeepPolicy(Policy):
    def __init__(
        self,
        state_dim: tuple,
        action_dim: tuple,
        has_continuous_action_space: bool = False,
        action_std_init: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super(DeepPolicy, self).__init__(
            state_dim, action_dim, has_continuous_action_space, action_std_init, device
        )

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )


def collect_trajectories_serial(
    env, agent: "Agent", num_timesteps_required: int
) -> int:
    state, _ = env.reset()
    current_ep_reward = 0
    current_num_timesteps = 0

    # Infos collected
    num_episodes: int = 0
    total_reward: float = 0.0
    num_timesteps: int = 0

    # buffer = BatchedRolloutBuffer(
    #     max_ep_len=agent.params.max_ep_len,
    #     batch_size=1,
    #     state_dim=agent.policy.state_dim,
    #     action_dim=1,  # ,agent.policy.action_dim,
    #     device=agent.device,
    # )

    while True:
        (
            action,
            action_stored,
            past_state,
            action_logprob,
            state_val,
        ) = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)

        # store the transition in buffer
        agent.buffer.batch_add(
            states=past_state.unsqueeze(0),
            actions=action_stored.unsqueeze(0),
            logprobs=action_logprob.unsqueeze(0),
            rewards=torch.tensor([[reward]], dtype=torch.float32, device=agent.device),
            state_values=state_val.unsqueeze(0),
            is_terminals=torch.tensor([[done]], dtype=bool, device=agent.device),
        )

        num_timesteps += 1
        current_num_timesteps += 1
        current_ep_reward += reward

        if done or current_num_timesteps >= agent.params.max_ep_len:
            num_episodes += 1
            total_reward += current_ep_reward
            # print(
            #     "  - Episode finished after {} timesteps".format(current_num_timesteps)
            # )
            if num_timesteps >= num_timesteps_required:
                break
            state, _ = env.reset()
            current_num_timesteps = 0
            current_ep_reward = 0
    # print(
    #     "  -> Collected {} episodes over {} timesteps (Avg reward: {})".format(
    #         num_episodes, num_timesteps, total_reward / num_episodes
    #     )
    # )
    return num_timesteps, num_episodes, total_reward


@dataclass
class TrainingParameters:
    # ========== Scheduling =========== #
    # Maximum number of timesteps per episode
    max_ep_len: int

    # Maximum number of timesteps in the whole training process
    max_training_timesteps: int

    # Number of timesteps to wait before updating the policy
    update_timestep: int

    # =========== Learning =========== #
    # Learning rate for actor
    lr_actor: float

    # Learning rate for critic
    lr_critic: float


@dataclass
class PPOTrainingParameters(TrainingParameters):
    # =========== Learning =========== #
    # Number of epochs to update the policy for
    K_epochs: int

    # Discount factor
    gamma: float

    # Epsilon for clipping
    eps_clip: float

    # =========== Continuous =========== #
    # Starting std for continuous action distribution (Multivariate Normal)
    action_std_init: Optional[float] = None


class Agent:
    def __init__(
        self,
        policy_builder: Callable[[], Policy],
        collect_trajectory_fn: callable,
        params: TrainingParameters,
        device: torch.device = torch.device("cpu"),
    ):
        self.device: torch.device = device
        self.params: TrainingParameters = params
        self.policy_builder: Callable[[], Policy] = policy_builder
        self.policy: Policy = policy_builder().to(self.device)
        self.collect_trajectory_fn: callable = collect_trajectory_fn

        self.buffer = BatchedRolloutBuffer(
            max_ep_len=400,
            batch_size=1,
            state_dim=self.policy.state_dim,
            action_dim=1,  # ,self.policy.action_dim,
            device=self.device,
        )

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Selects an action to take given a state

        Args:
            state (torch.Tensor): The state to calculate the action for (Shape: (batch_size, state_dim))

        Returns:
            action (torch.Tensor): The action to take (Shape: (batch_size, action_dim))
            action_tensor (torch.Tensor): The action to store (Shape: (batch_size, action_dim))
            past_state (torch.Tensor): The state to calculate the action for (Shape: (batch_size, state_dim))
            action_logprob (torch.Tensor): The log probability of the action (Shape: (batch_size, 1))
            state_val (torch.Tensor): The value of the state (Shape: (batch_size, 1))
        """

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_tensor, action_logprob, state_val = self.policy_old.act(state)

        if self.has_continuous_action_space:
            action = action_tensor.detach().cpu().numpy().flatten()
        else:
            action = action_tensor.item()

        return action, action_tensor, state, action_logprob, state_val

    @abstractmethod
    def save(self, checkpoint_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        raise NotImplementedError


class PPO(Agent):
    def __init__(
        self,
        policy_builder: Callable[[], Policy],
        collect_trajectory_fn: callable,
        params: PPOTrainingParameters,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(policy_builder, collect_trajectory_fn, params, device)

        # ============ Continuous ============ #
        self.has_continuous_action_space = self.policy.has_continuous_action_space
        if self.has_continuous_action_space:
            self.action_std = self.params.action_std_init
            self.policy.action_std = self.action_std

        # Old policy to calculate the ratio
        self.policy_old = policy_builder().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print(
                "WARNING : Calling PPO::set_action_std() on discrete action space policy"
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print(
            "--------------------------------------------------------------------------------------------"
        )

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print(
                    "setting actor output action_std to min_action_std : ",
                    self.action_std,
                )
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print(
                "WARNING : Calling PPO::decay_action_std() on discrete action space policy"
            )

        print(
            "--------------------------------------------------------------------------------------------"
        )

    def update(self):
        from toolbox.printing import debug

        (
            old_states,
            old_actions,
            old_logprobs,
            old_rewards,
            old_state_values,
            old_is_terminals,
        ) = self.buffer.sample(
            0
        )  # TODO: Change this to batch_size
        # debug(self.buffer.linear_buffer.num_entries)

        # Monte Carlo estimate of returns
        # Reverse the tensors
        old_rewards = old_rewards.flip(0)
        old_is_terminals = old_is_terminals.flip(0)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(old_rewards, old_is_terminals):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.params.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # debug(old_actions)
        # debug(old_logprobs)
        # debug(old_state_values)
        # debug(old_states)
        # debug(rewards)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.params.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.params.eps_clip, 1 + self.params.eps_clip)
                * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path: str) -> None:
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str) -> None:
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )

    def train(
        self,
        env,
        print_freq: int,
        log_freq: int,
        save_model_freq: int,
        log_f_name: str,
        checkpoint_path: str,
    ):
        # Training stuff
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": self.params.lr_actor},
                {
                    "params": self.policy.critic.parameters(),
                    "lr": self.params.lr_critic,
                },
            ]
        )
        self.MseLoss = nn.MSELoss()

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print(
            "============================================================================================"
        )

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write("episode,timestep,reward\n")

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while time_step <= self.params.max_training_timesteps:
            # collect trajectories
            # print("Collecting trajectories...")
            num_timesteps, num_episodes, total_reward = self.collect_trajectory_fn(
                env, self, self.params.update_timestep
            )
            time_step += num_timesteps
            print_running_reward += total_reward
            print_running_episodes += num_episodes
            log_running_reward += total_reward
            log_running_episodes += num_episodes
            i_episode += num_episodes

            # update PPO agent
            self.update()

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(
                    "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                        i_episode, time_step, print_avg_reward
                    )
                )

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                print("saving model at : " + checkpoint_path)
                self.save(checkpoint_path)
                print("model saved")
                print(
                    "Elapsed Time  : ",
                    datetime.now().replace(microsecond=0) - start_time,
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )

                # # break; if the episode is over
                # if done:
                #     break

        log_f.close()
        env.close()

        # print total training time
        print(
            "============================================================================================"
        )
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print(
            "============================================================================================"
        )
