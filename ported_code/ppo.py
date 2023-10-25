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
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


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
            cov_mat = torch.diag_embed(action_var).to(device)
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
                (self.action_dim,), new_action_std * new_action_std
            ).to(device)
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

    while True:
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)
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
        self.buffer = RolloutBuffer()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Selects an action to take given a state

        Args:
            state (torch.Tensor): The state to calculate the action for (Shape: (batch_size, state_dim))

        Returns:
            action (torch.Tensor): The action to take (Shape: (batch_size, action_dim))
        """

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

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
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.params.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

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

            # state, _ = env.reset()
            # current_ep_reward = 0

            # for t in range(1, max_ep_len + 1):
            #     # select action with policy
            #     action = ppo_agent.select_action(state)
            #     state, reward, done, _, _ = env.step(action)

            #     # saving reward and is_terminals
            #     ppo_agent.buffer.rewards.append(reward)
            #     ppo_agent.buffer.is_terminals.append(done)

            #     time_step += 1
            #     current_ep_reward += reward

            #     # update PPO agent
            #     if time_step % update_timestep == 0:
            #         ppo_agent.update()

            #     # if continuous action space; then decay action std of ouput action distribution
            #     if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            #         ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

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
