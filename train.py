# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import ParallelEnv
from rl_autoschedular.model import HiearchyModel as Model
import torch
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

os.environ["PATH"] = f"{os.environ['SCRATCH']}/gcc/bin:{os.environ['SCRATCH']}/lld/bin:{os.environ['PATH']}"
os.environ["LD_LIBRARY_PATH"] = f"{os.getenv('SCRATCH')}/gcc/lib64:{os.getenv('LD_LIBRARY_PATH')}"
os.environ["CXX"] = f"{os.getenv('SCRATCH')}/gcc/bin/g++"

# print_info("$CXX = ",os.getenv("CXX"))

# Set environments
env = ParallelEnv(
    num_env=1,
    reset_repeat=1,
    step_repeat=1
)
eval_env = ParallelEnv(
    num_env=1,
    reset_repeat=1,
    step_repeat=1
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
neptune_logs = init_neptune(['hierchical', 'sparse_reward'] + cfg.tags) if cfg.logging else None

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

    torch.save(model.state_dict(), 'models/ppo_model.pt')

    if step % 5 == 0:
        evaluate_benchmark(
            model=model,
            env=eval_env,
            device=device,
            neptune_logs=neptune_logs
        )

        if cfg.logging:
            neptune_logs["params"].upload_files(['models/ppo_model.pt'])


# Stop logs if enabled
if cfg.logging:
    neptune_logs.stop()

    print('\n','-'*30)
    with open("neptune.out","w") as f:
        f.write(str(neptune_logs))
        print("neptune has been logged")
    
    for i in ['train/final_speedup', 'train/cummulative_reward', 'train/policy_loss', 'train/value_loss', 'eval/final_speedup']:
        print_info(i,':',neptune_logs[i])

    print('\n',"-"*30)

print_info('Training ended ... ')
