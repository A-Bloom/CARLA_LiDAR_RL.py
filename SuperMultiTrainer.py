from sbx import DroQ
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common import utils
from pathlib import Path
from datetime import datetime
from sys import platform
from subprocess import Popen
import os
from Envs.Env1 import CarEnv as Env1
from Envs.Env2 import CarEnv as Env2
from Envs.Env3 import CarEnv as Env3
from Envs.Env4 import CarEnv as Env4
from Envs.Env5 import CarEnv as Env5
from Envs.Env6 import CarEnv as Env6

envs = ["Env4", "Env5", "Env6"]

timestamp = datetime.now()
folder_name = ("Output/" + Path(__file__).stem + timestamp.strftime("_%m_%d_%H_%M"))
log_dir = f"{folder_name}/TBLogs"
os.makedirs(log_dir, exist_ok=True)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))
device = "auto"

cycles = 50
timeSteps = 50000

if platform == "linux":
    os.popen(f"python -m tensorboard.main --logdir={log_dir}")
    print(f"Opening TensorBoard at {log_dir}")
    device = "cuda:1"
elif platform == "win32":
    Popen(f"py -m tensorboard.main --logdir={log_dir}", creationflags=0x00000008)
    print(f"Opening TensorBoard at {log_dir}")


for names in envs:

    env = locals()[names]()

    print(f"Initializing {env.name()}")

    lr = 0.001
    while lr <= 0.091:

        model = DroQ('MlpPolicy', env, verbose=0, learning_rate=0.001, learning_starts=10000, tensorboard_log=log_dir,
                     device=device, gradient_steps=20, policy_delay=20, dropout_rate=0.01, layer_norm=True)
        tb_dir = f"{env.name()}_{int(lr*1000)}"
        model_dir = f"{folder_name}/{env.name()}/{int(lr*1000)}"
        print(f"Learning rate of {lr}")

        for i in range(1, cycles + 1):

            print("Beginning cycle " + str(i) + " of " + str(cycles))
            model.learn(total_timesteps=timeSteps, reset_num_timesteps=False, tb_log_name=tb_dir)
            os.makedirs(model_dir, exist_ok=True)
            model.save(f"{model_dir}/{i*timeSteps}")

        lr = lr + 0.03
        del model

    env.close()

#-------------------------------------------------------------------

envs = ["Env1", "Env2", "Env3", "Env4", "Env5", "Env6"]

timestamp = datetime.now()
folder_name = ("Output/" + Path(__file__).stem + timestamp.strftime("_%m_%d_%H_%M"))
log_dir = f"{folder_name}/TBLogs"
os.makedirs(log_dir, exist_ok=True)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))
device = "auto"

cycles = 50
timeSteps = 5000

if platform == "linux":
    os.popen(f"python -m tensorboard.main --logdir={log_dir}")
    print(f"Opening TensorBoard at {log_dir}")
    device = "cuda:1"
elif platform == "win32":
    Popen(f"py -m tensorboard.main --logdir={log_dir}", creationflags=0x00000008)
    print(f"Opening TensorBoard at {log_dir}")


for names in envs:

    env = locals()[names]()

    print(f"Initializing {env.name()}")

    lr = 0.001
    while lr <= 0.091:

        model = PPO('MlpPolicy', env, verbose=0, learning_rate=lr, tensorboard_log=log_dir,device=device)
        tb_dir = f"{env.name()}_{int(lr*1000)}"
        model_dir = f"{folder_name}/{env.name()}/{int(lr*1000)}"
        print(f"Learning rate of {lr}")

        for i in range(1, cycles + 1):

            print("Beginning cycle " + str(i) + " of " + str(cycles))
            model.learn(total_timesteps=timeSteps, reset_num_timesteps=False, tb_log_name=tb_dir)
            os.makedirs(model_dir, exist_ok=True)
            model.save(f"{model_dir}/{i*timeSteps}")

        lr = lr + 0.03
        del model

    env.close()

#--------------------------------------------------------------

envs = ["Env1", "Env2", "Env3", "Env4", "Env5", "Env6"]

timestamp = datetime.now()
folder_name = ("Output/" + Path(__file__).stem + timestamp.strftime("_%m_%d_%H_%M"))
log_dir = f"{folder_name}/TBLogs"
os.makedirs(log_dir, exist_ok=True)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))
device = "auto"

cycles = 50
timeSteps = 5000

if platform == "linux":
    os.popen(f"python -m tensorboard.main --logdir={log_dir}")
    print(f"Opening TensorBoard at {log_dir}")
    device = "cuda:1"
elif platform == "win32":
    Popen(f"py -m tensorboard.main --logdir={log_dir}", creationflags=0x00000008)
    print(f"Opening TensorBoard at {log_dir}")


for names in envs:

    env = locals()[names]()

    print(f"Initializing {env.name()}")

    lr = 0.001
    while lr <= 0.091:

        model = A2C('MlpPolicy', env, verbose=0, learning_rate=lr, tensorboard_log=log_dir,device=device)
        tb_dir = f"{env.name()}_{int(lr*1000)}"
        model_dir = f"{folder_name}/{env.name()}/{int(lr*1000)}"
        print(f"Learning rate of {lr}")

        for i in range(1, cycles + 1):

            print("Beginning cycle " + str(i) + " of " + str(cycles))
            model.learn(total_timesteps=timeSteps, reset_num_timesteps=False, tb_log_name=tb_dir)
            os.makedirs(model_dir, exist_ok=True)
            model.save(f"{model_dir}/{i*timeSteps}")

        lr = lr + 0.03
        del model

    env.close()

#------------------------------------------------------------------------------

envs = ["Env4", "Env5", "Env6"]

timestamp = datetime.now()
folder_name = ("Output/" + Path(__file__).stem + timestamp.strftime("_%m_%d_%H_%M"))
log_dir = f"{folder_name}/TBLogs"
os.makedirs(log_dir, exist_ok=True)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))
device = "auto"

cycles = 50
timeSteps = 5000

if platform == "linux":
    os.popen(f"python -m tensorboard.main --logdir={log_dir}")
    print(f"Opening TensorBoard at {log_dir}")
    device = "cuda:1"
elif platform == "win32":
    Popen(f"py -m tensorboard.main --logdir={log_dir}", creationflags=0x00000008)
    print(f"Opening TensorBoard at {log_dir}")


for names in envs:

    env = locals()[names]()

    print(f"Initializing {env.name()}")

    lr = 0.001
    while lr <= 0.091:

        model = SAC('MlpPolicy', env, verbose=0, learning_rate=lr, tensorboard_log=log_dir,device=device)
        tb_dir = f"{env.name()}_{int(lr*1000)}"
        model_dir = f"{folder_name}/{env.name()}/{int(lr*1000)}"
        print(f"Learning rate of {lr}")

        for i in range(1, cycles + 1):

            print("Beginning cycle " + str(i) + " of " + str(cycles))
            model.learn(total_timesteps=timeSteps, reset_num_timesteps=False, tb_log_name=tb_dir)
            os.makedirs(model_dir, exist_ok=True)
            model.save(f"{model_dir}/{i*timeSteps}")

        lr = lr + 0.03
        del model

    env.close()