from stable_baselines3 import PPO
from stable_baselines3.common import utils
from pathlib import Path
from datetime import datetime
from sys import platform
from subprocess import Popen
import os
from Env1 import CarEnv as Env1
from Env2 import CarEnv as Env2
from Env3 import CarEnv as Env3

envs = ["Env1", "Env2", "Env3"]

timestamp = datetime.now()
folder_name = ("Output/" + Path(__file__).stem + timestamp.strftime("_%m_%d_%H_%M"))
log_dir = f"{folder_name}/TBLogs"
os.makedirs(log_dir, exist_ok=True)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

cycles = 10
timeSteps = 5000

if platform == "win32":
    os.popen(f"py -m tensorboard.main --logdir={log_dir}")
    print(f"Opening TensorBoard at {log_dir}")
elif platform == "linux":
    Popen(f"python -m tensorboard.main --logdir={log_dir}", creationflags=0x00000008)
    print(f"Opening TensorBoard at {log_dir}")


for names in envs:

    env = locals()[names]()

    print(f"Initializing {env.name()}")

    lr = 0.001
    while lr < 0.102:

        model = PPO('MlpPolicy', env, verbose=0, learning_rate=lr, tensorboard_log=log_dir)
        tb_dir = f"{env.name()}_{int(lr*1000)}"
        model_dir = f"{folder_name}/{env.name()}/{int(lr*1000)}"
        print(f"Learning rate of {lr}")

        for i in range(1, cycles + 1):

            print("Beginning cycle " + str(i) + " of " + str(cycles))
            model.learn(total_timesteps=timeSteps,  reset_num_timesteps=False, tb_log_name=tb_dir)
            os.makedirs(model_dir, exist_ok=True)
            model.save(f"{model_dir}/{i*timeSteps}")

        lr = lr + 0.01
        del model

    env.close()
