# Load environment variables
from typing import Optional,IO
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import ParallelEnv

from rl_autoschedular.evaluation import evaluate_code_with_bindings_and_timeout, evaluate_code_with_cmd_and_timeout

from rl_autoschedular.model import HiearchyModel as Model
import torch
import numpy as np
from rl_autoschedular import config as cfg
from utils.log import print_info
# from rl_autoschedular.ppo import evaluate_benchmark

def evaluate_benchmark(
    model: Model,
    env: ParallelEnv,
    f: Optional[IO[str]] = None
):
    """Evaluate the benchmark using the model (save the results to a csv).

    Args:
        model (Model): The model to use.
        env (ParallelEnv): The environment to use.
        f (Optional[IO[str]]): A writable file-like object to log results into. Defaults to None.
    """
    # NOTE: Only using one environment

    for i, (bench_name, benchmark_data) in enumerate(env.envs[0].benchmarks_data):
        if cfg.data_format == 'json' and "bench" not in benchmark_data.bench_name:
            op_tag = benchmark_data.operation_tags[-1]
            instance = benchmark_data.operations[op_tag].raw_operation
        else:
            instance = bench_name

        print(f"Operation ({i}): {instance}")
        if instance == "linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins (%input, %filter: tensor<256x3x260x260xf32>, tensor<64x3x3x3xf32>) outs (%init: tensor<256x64x129x129xf32>) -> tensor<256x64x129x129xf32>":
            print("skipped")
            continue

        # Reset the environment with the specific operation
        state, obs = env.reset(i)

        while True:
            x = obs[0]

            with torch.no_grad():
                # Select the action using the model
                action, _, _, _ = model.sample(x,greedy=True)

            # Apply the action and get the next state
            next_obs, _, terminated, next_state, final_state = env.step(state, action, fail_transform=False)

            done = terminated[0]
            final_state = final_state[0]
            if done and final_state is not None:
                speedup_metric = final_state.root_exec_time / final_state.exec_time

                print('Base execution time:', final_state.root_exec_time, 's')
                print('New execution time:', final_state.exec_time, 's')
                print('Speedup:', speedup_metric)

                if f is not None : f.write(
                    f"{bench_name:<20},{final_state.root_exec_time},{final_state.exec_time},{speedup_metric}\n"
                )
                
                break

            state = next_state
            obs = next_obs

def init_env_from_mlir(*file_paths: str, save_set: Optional[bool] = False) -> ParallelEnv:
    """
    Initialize the environment from one or more MLIR files.

    Args:
        *file_paths (str): One or more file paths that point to valid MLIR code which should contain the nn model and should be directly runnable.
        save_set (bool): whether we should save the resulting json structure used in Env class. If True, it gets saved in "evaluate_set.json". Defaults to False.

    Returns:
        ParallelEnv: env object that uses the nn models in its benchmark_data.

    Raise:
        ValueError: if any file is not able to be opened.
        AssertionError: if any nn code is not runnable (no main functions) or fails to run.
    """
    json_data = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            code = f.read()

        if not code:
            raise ValueError(f"Failed to read MLIR code from file: {file_path}")

        execs = []
        for _ in range(3):
            execution_time, passed = evaluate_code_with_bindings_and_timeout(code,timeout=None)
            
            assert passed and execution_time is not None, f"evaluating the Mlir code failed for {file_path}"
            
            execs.append(execution_time)

        exec_time = np.median(execs)

        file_name = file_path.split("/")[-1] if file_path.split("/") else file_path
        file_name = file_name.split(".")[0] if file_name.split(".") else file_name

        json_data.append(
            (
                f"Model-{file_name}",
                {
                    "transform_wrapped_operation": code,
                    "execution_time": exec_time
                }
            )
        )

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

print_info('Finish imports')

# Set environment
# env = init_env_from_mlir(
#     # "./benchs/DenseNet-bench.mlir",
#     "./benchs/MobileNetV2-bench.mlir","./benchs/ResNet-bench.mlir","./benchs/VGG-bench.mlir",
#     save_set=True
# )

import json
env = ParallelEnv(
    # env_json_data = list(json.load(open("eval.json")).items())
    env_json_data = list(json.load(open("evaluate_set.json")).items())
)

print_info('Env build ...')

# NOTE: using only one environment
print_info(f'tmp_file = {env.envs[0].tmp_file}')

# Print configuration
print_info('Configuration:')
print_info(cfg)

# import better_exceptions
# better_exceptions.hook()

model_checkpoint = "models/ppo_model_MLIR-140-1.pt"

# Set model
model = Model()
print_info('input_dim:', model.input_dim)

model.load_state_dict(torch.load(model_checkpoint))

file = "eval-models-140-no-vect.csv"

print_info(file)

with open(file,"w") as f:
    
    f.write("Name,base,new,speedup\n")
    
    evaluate_benchmark(
        model=model,
        env=env,
        f = f
    )
