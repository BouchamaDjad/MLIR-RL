# Load environment variables
from typing import Optional
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import ParallelEnv

from rl_autoschedular.evaluation import evaluate_code_with_cmd_and_timeout

from rl_autoschedular.model import HiearchyModel as Model
import torch
import numpy as np
from rl_autoschedular import config as cfg
from utils.log import print_info
from rl_autoschedular.ppo import evaluate_benchmark

def init_env_from_mlir(file_path: str, save_set: Optional[bool] = False) -> ParallelEnv:
    """
    Initialize the environment from a MLIR file.

    Args:
        file_path (str): file_path that points to valid MLIR code which should contain the nn model and should be directly runnable.
        save_set (bool): whether we should save the resulting json structure used in Env class. If True, it gets saved in "evaluate_set.json". Defaults to False.

    Returns:
        ParallelEnv: env object that uses the nn model in its benchmark_data.

    Raise:
        ValueError: if file is not able to be opened.
        AssertionError: if the nn code is not runnable (no main functions) or fails to run.
        
    """
    with open(file_path, 'r') as f:
        code = f.read()

    if not code:
        raise ValueError(f"Failed to read MLIR code from file: {file_path}")

    execs = []
    for _ in range(3):
        execution_time,passed = evaluate_code_with_cmd_and_timeout(code, file_path, timeout=300)

        assert passed and execution_time is not None,"evaluating the Mlir code failed"

        execs.append(execution_time)

    exec_time = np.median(execs)
    # print(exec_time)

    file_name = file_path.split("/")[-1] if file_path.split("/") else file_path
    file_name = file_name.split(".")[0] if file_name.split(".") else file_name
    # print(file_name)

    json_data = [
        (
            f"bench-{file_name}",
            {
                "transform_wrapped_operation": code,
                "execution_time": exec_time
            }
        )
    ]

    if save_set:
        with open("evaluate_set.json", "w") as f:
            import json
            json.dump(dict(json_data), f, indent=4)

    env = ParallelEnv(
        num_env=1,
        reset_repeat=1,
        step_repeat=1,
        env_json_data=json_data,
    )

    return env

device = torch.device("cpu")

# TODO: make them env passed arguments
benchmark_model = ""
model_checkpoint = "models/ppo_model.pt"


print_info('Finish imports')

# Set environment
env = init_env_from_mlir(benchmark_model)
print_info('Env build ...')

# NOTE: using only one environment
print_info(f'tmp_file = {env.envs[0].tmp_file}')

# Print configuration
print_info('Configuration:')
print_info(cfg)

# import better_exceptions
# better_exceptions.hook()

# Set model
model = Model()
print_info('input_dim:', model.input_dim)

model.load_state_dict(torch.load(model_checkpoint))

evaluate_benchmark(
    model=model,
    env=env,
    device=device,
    neptune_logs=None
)
