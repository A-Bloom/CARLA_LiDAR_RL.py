from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.env_checker import check_env
from subprocess import Popen
import os
from sys import platform
from Env3 import CarEnv
from pathlib import Path
from datetime import datetime

env = CarEnv()

print("Initializing " + env.name())

models_dir = f"Output/Models/{Path(__file__).stem}"
log_dir = f"Output/Logs/{Path(__file__).stem}"
timestamp = datetime.now()
file_name = (env.name() + timestamp.strftime("_%m_%d_%H_%M"))
os.makedirs(log_dir, exist_ok=True)
device = "auto"

if platform == "linux":
    os.popen(f"python -m tensorboard.main --logdir={log_dir}/{file_name}_0")
    print(f"Opening TensorBoard at {log_dir}/{file_name}_0")
    device = "cuda:1"
elif platform == "win32":
    Popen(f"py -m tensorboard.main --logdir={log_dir}/{file_name}_0", creationflags=0x00000008)
    print(f"Opening TensorBoard at {log_dir}/{file_name}_0")

model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.001, tensorboard_log=log_dir, device=device)

cycles = 2
timeSteps = 10

#check_env(env)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))


for i in range(1, cycles + 1):
    print("Beginning cycle " + str(i) + " of " + str(cycles))
    model.learn(total_timesteps=timeSteps, reset_num_timesteps=False, tb_log_name=file_name)
    os.makedirs(models_dir, exist_ok=True)
    model.save(f"{models_dir}/{file_name}")

env.close()
