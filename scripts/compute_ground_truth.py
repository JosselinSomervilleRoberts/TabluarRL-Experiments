# Filter warnings
import warnings

warnings.filterwarnings("ignore")

import argparse

from envs.ground_truth import GroundTruth, compute_ground_truth, save_ground_truth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-MultiRoom-N6-v0",
        help="Name of the MDP.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_new/",
        help="Directory containing the MDPs.",
    )
    parser.add_argument(
        "--horizon", type=int, default=100, help="Horizon for value iteration."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="Discount factor for value iteration."
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Compute ground truth
    gt: GroundTruth = compute_ground_truth(
        args.env_name, args.data_dir, horizon=args.horizon, gamma=args.gamma
    )

    # Save ground truth
    save_ground_truth(args.env_name, gt)


if __name__ == "__main__":
    args = parse_args()
    main(args)
