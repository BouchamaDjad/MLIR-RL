# Load environment variables
import random
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import (
    ParallelEnv,
    train_eval_split
)
                                  
from rl_autoschedular.model import HiearchyModel as Model
import torch
import json
from tqdm import tqdm
from rl_autoschedular import config as cfg
from utils.log import print_info
from utils.neptune_utils import init_neptune
from rl_autoschedular.ppo import (
    collect_trajectory,
    ppo_update,
    evaluate_benchmark
)

# Set target device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print_info('Finish imports')

def reduce_dataset_size():
    with open(cfg.json_file, "r") as file:
        json_data = json.load(file)
        json_data = list(json_data.items())

    patterns = [(key,item) for key,item in json_data if "single" not in key and "bench" not in key.split("_")[0]]    
    singles = [(key,item) for key,item in json_data if "single" in key]

    synthesized = [(key,item) for key,item in json_data if "bench" in key.split("_")[0]]
    random.shuffle(synthesized)

    synth_len = cfg.dataset_length - (len(patterns) + len(singles))
    
    if synth_len > 0:
        synthesized = synthesized[:synth_len]
    
    json_data = patterns + synthesized + singles
    
    random.shuffle(json_data)

    return json_data

# Set environments
if cfg.data_format == "json":
    if cfg.dataset_length != 0:
        reduced_set = reduce_dataset_size()
    else:
        reduced_set = None
        print_info("The full dataset will be used")

    if cfg.train_eval_split:
        env,eval_env = train_eval_split(reduced_set, eval_size=cfg.train_eval_split)
    
    else:
        env = ParallelEnv(
            num_env=1,
            reset_repeat=1,
            step_repeat=1
        )

        eval_env = ParallelEnv(
            num_env=1,
            reset_repeat=1,
            step_repeat=1,
            # In case you will train on single-operations evaluation set (mentioned in the paper)
            # env_json_data = list(json.load(open("data/nn/eval_operations.json")).items())
        ) 

else:
    env = ParallelEnv(
        num_env=1,
        reset_repeat=1,
        step_repeat=1
    )

    eval_env = ParallelEnv(
        num_env=1,
        reset_repeat=1,
        step_repeat=1,
    )

print_info('Env build ...')
# NOTE: using only one environment
print_info(f'tmp_file = {env.envs[0].tmp_file}')

# Print configuration
print_info('Configuration:')
print_info(cfg)

# Set model
model = Model()
print_info('input_dim:', model.input_dim)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)

# Set neptune logs if enabled
neptune_logs = init_neptune(['type_op','action-history'] + cfg.tags) if cfg.logging else None

# get run id 
run_id = neptune_logs["sys/id"].fetch() if cfg.logging else ""
print_info(f"Run id: {run_id}")

# Start training
print_info('Start training ... ')
tqdm_range = tqdm(range(cfg.nb_iterations), desc='Main loop')
for step in tqdm_range:

    trajectory = collect_trajectory(
        cfg.len_trajectory,
        model,
        env,
        device=device,
        neptune_logs=neptune_logs
    )

    loss = ppo_update(
        trajectory,
        model,
        optimizer,
        ppo_epochs=cfg.ppo_epochs,
        ppo_batch_size=cfg.ppo_batch_size,
        device=device,
        entropy_coef=cfg.entropy_coef,
        neptune_logs=neptune_logs
    )

    torch.save(model.state_dict(), f'models/ppo_model_{run_id}.pt')

    if step % 5 == 0:
        evaluate_benchmark(
            model=model,
            env=eval_env,
            device=device,
            neptune_logs=neptune_logs
        )

        if cfg.logging:
            neptune_logs["params"].upload_files([f'models/ppo_model_{run_id}.pt'])


# Stop logs if enabled
if cfg.logging:
    neptune_logs.stop()

print_info('Training ended ... ')
