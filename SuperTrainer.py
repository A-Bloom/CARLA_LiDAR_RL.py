from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.evaluation import evaluate_policy
import os
import matplotlib.pyplot as plt
from Env1 import CarEnv as Env1
from Env2 import CarEnv as Env2
import time

envs = [Env1(), Env2()]


for env in envs:

    lr = 0.001
    while lr < 1:

        model = PPO('MlpPolicy', env, verbose=0, learning_rate=lr)

        cycles = 10
        timeSteps = 500

        #  check_env(env)
        print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))
        print("Beginning cycle 1")

        model.learn(total_timesteps=timeSteps)

        models_dir = f"Output/models/"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model.save(f"{models_dir}/{int(lr*1000)}")

        for i in range(cycles - 1):
            print("Beginning cycle " + str(i + 2))
            model.learn(total_timesteps=timeSteps)
            model.save(f"{models_dir}/{int(lr*1000)}")

        lr = lr + 0.005

    env.close()
