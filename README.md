# TabularRL-Experiments
Experiments on RL exploration on tabular environments to generalize concepts in hte Madrona engine


## Compute ground truth
To get the ground truth infos about an environment, run `python3 scripts/get_ground_truth.py --env <env_name>`. This will generate a folder in `output/envs/<env_name>` with the following files:
- `v.npz`: Optimal V values (tabular)
- `v_grid.npz`: Optimal V values (grid)
- `v_grid.png`: Heatmap of V values (grid)
- `q.npz`: Optimal Q values.
- `q_grid.npz`: Optimal Q values (grid)
- `path.npz`: Optimal path.
- `policy.png`: Directions to take at each state *(only displays the forward action)* on top of the heatmap of V values.
- `mapping.pkl`: Mapping of tabular state (`int`) to game state (`GameState`).

You can generate the ground truth for all environments by running `sh scripts/computer_all_ground_truths.sh`.