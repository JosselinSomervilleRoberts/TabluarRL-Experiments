import os

import torch
import gym

from ppo import (
    PPO,
    PPOTrainingParameters,
    DeepPolicy,
    collect_trajectories_serial,
    TabularPolicy,
    collect_trajectories_parallel,
)

from envs.tabular_world import TabularWorld


################################## set device ##################################

print(
    "============================================================================================"
)


# set device to cpu or cuda
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print(
    "============================================================================================"
)


print(
    "============================================================================================"
)

# Parse from command line env_type
import argparse

parser = argparse.ArgumentParser(description="Process the environment type.")

# Add the env_type argument
# The 'dest' parameter is optional. It defines the name of the attribute where the parsed value will be stored.
# If 'dest' is not provided, argparse will use the option string ('--env_type' in this case) to determine the name.
parser.add_argument(
    "--env_type",
    type=str,
    required=True,
    help="The type of environment (e.g., tabular, gym)",
)

# Parse the arguments
args = parser.parse_args()
env_type = args.env_type


if env_type == "gym":
    env_name = "CartPole-v1"
    has_continuous_action_space = False
    action_std = None
    random_seed = 0  # set random seed if required (0 = no random seed)
    env = gym.make(env_name)
elif env_type == "tabular":
    env_name = "MiniGrid-DoorKey-8x8-OpenDoorsPickupShaped-v0"
    data_dir = "data_new/"
    file_name = f"{data_dir}/{env_name}/consolidated.npz"
    random_seed = 0  # set random seed if required (0 = no random seed)
    env = TabularWorld(file_name, num_worlds=4096, device=device)
else:
    raise ValueError(f"Environment {env_type} not supported.")

print("training environment name : " + env_name)


###################### logging ######################

#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir = log_dir + "/" + env_name + "/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)


#### create new log file for each run
log_f_name = log_dir + "/PPO_" + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)

#####################################################


################### checkpointing ###################

run_num_pretrained = (
    0  #### change this to prevent overwriting weights in same env_name folder
)

directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + "/" + env_name + "/"
if not os.path.exists(directory):
    os.makedirs(directory)


checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(
    env_name, random_seed, run_num_pretrained
)
print("save checkpoint path : " + checkpoint_path)

#####################################################

if env_type == "gym":
    params = PPOTrainingParameters(
        max_training_timesteps=int(1e5),
        max_ep_len=400,
        update_timestep=1600,
        update_batch_size=1600,
        K_epochs=40,
        eps_clip=0.2,
        gamma=0.99,
        lr_actor=0.0003,
        lr_critic=0.001,
        state_type=torch.float32,
    )

    def policy_factory():
        return DeepPolicy(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
            if has_continuous_action_space
            else env.action_space.n,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std,
            device=device,
        )

    collect_trajectory_fn = collect_trajectories_serial
else:
    params = PPOTrainingParameters(
        max_training_timesteps=int(1e8),
        max_ep_len=80,
        update_timestep=env.num_worlds * 80,
        K_epochs=40,
        eps_clip=0.2,
        gamma=0.99,
        lr_actor=0.0003,
        lr_critic=0.001,
        update_batch_size=env.num_worlds * 80,
        state_type=torch.int32,
    )

    def policy_factory():
        return TabularPolicy(
            num_states=env.num_states, num_actions=env.num_actions, device=device
        )

    collect_trajectory_fn = collect_trajectories_parallel


# initialize a PPO agent
ppo_agent = PPO(
    policy_builder=policy_factory,
    collect_trajectory_fn=collect_trajectory_fn,
    params=params,
    device=device,
)

ppo_agent.train(
    env=env,
    print_freq=1,
    log_freq=1,
    save_model_freq=20000,
    log_f_name=log_f_name,
    checkpoint_path=checkpoint_path,
)
