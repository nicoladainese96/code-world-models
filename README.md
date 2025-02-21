# Code World Models

Code for the "Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search" paper published at [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html).

See webpage at https://sites.google.com/view/code-world-models/home

## Installation

To install the required packages, run the following command:

```bash
conda env create -f environment.yml --name codeworldmodels
```

Note that this also requires MuJoCo to be installed. To do this, follow the instructions at [this page](https://github.com/pytorch/rl/blob/main/knowledge_base/MUJOCO_INSTALLATION.md).

For the RTFM environment, you will need to install a customized version to allow for the stationary and deterministic environment. To do this, from the project root directory, run the following commands:

```bash
cd RTFM
pip install -e .
```

## Experiments

For each experiment you will need to specify which underlying LLM you want to use. If 'gpt' is in the model name, you will also need to create a folder named 'openai' in the root directory and include your API key and organization in two files named 'openai_key' and 'openai_org' respectively. Otherwise, the model will be loaded from the Transformers library. The model needs to be given to each experiment script using the `--model` argument.

The `--budget` argument is used to specify the number of LLM calls for all methods. This also needs to be specified for each experiment.

### APPS

To run the APPS experiment, first download the dataset using the following command:

```bash
./sh_scripts/download_apps_data.sh
```

Then, to run GIF-MCTS, use the following command:

```bash
python3 src/experiments/run_mcts_apps_all_prob.py --idx 0 --total_tasks 100 
```

The script is designed to only run a portion of the tasks to allow for parallelization. This particular command will run the first 100 tasks. To run the remaining tasks, simply change the `idx` (the starting index) and `total_tasks` (the number of tasks to run) arguments.

To run WorldCoder with the APPS dataset, use the following command:

```bash
python3 src/experiments/run_world_coder_apps.py --idx 0 --total_tasks 100
```

For the Zero-shot CoT baseline, use the following command:

```bash
python3 src/experiments/run_generate_solutions_apps.py
```

After the all json files for the solutions are generated, use the following command to evaluate the solutions:

```bash
python3 src/experiments/eval_solutions_apps.py --save save_path # Where the evaluated data is loaded from and results saved to.
```

### CWMB

All scripts for the CWMB require a replay buffer of transitions. We provide the dataset we used for all experiments for the paper in the `data/replay_buffers` folder, as these can be a source of stochasticity in the experiments. If you wish to gather a new replay buffer, you can use the following python code:

```python
from src.replay_buffer import fill_replay_buffer
import gymnasium as gym

env = gym.make('CartPole-v1') # or any other gym environment
fill_replay_buffer(env, capacity=n, file_path='data/replay_buffers/gymnasium_envs/env_name', buffer_name=train_buffer)
```

All scripts for the CWMB accept either a `--idx` argument to specify the index of the environment to run (useful for parallelization) or a `--env` argument to specify the name of the environment to run. The `--budget` argument is used to specify the number of LLM calls for all methods.
The environment names are the same as the ones in the gymnasium library, and a list can be found in the `data/prompts/gymnasium_envs` folder by listing all the folders in that directory. Additionally, `--env rtfm` can be used to run the RTFM environment.

```bash
python3 src/experiments/run_mcts_cwm.py --idx 0 # runs the first environment in the CWMB dataset
```

or

```bash
python3 src/experiments/run_mcts_cwm.py --env CartPole-v1 # runs the CartPole-v1 environment
```

To run WorldCoder with the CWMB dataset, use the following command:

```bash
python3 src/experiments/run_world_coder_cwm.py --idx 0 # or --env env_name
```

After the CWMs have been generated, use the following command to compute the return when planning with the CWM:

```bash
python src/experiments/eval_cwm_planning.py --save_dir save_path # Where the CWM is stored
```

Additionally, the `--real_env True` argument can be added to the previous command to calculate the planning results on the real environment.

### Additional Experiments

To replicate the inference speed experiment from Appendix C, use the following command:

```bash
python3 src/experiments/eval_inference_speed_llm.py --model model_name --n_transitions n_transitions
```

where `model_name` is the name of the model and `n_transitions` is the number of transitions to sample from the replay buffer and to compare the inference speed on.
