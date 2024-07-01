from stable_baselines3 import PPO
from stable_baselines3.common import utils
from pathlib import Path
from datetime import datetime
import os
import matplotlib.pyplot as plt
from Env1 import CarEnv as Env1
from Env2 import CarEnv as Env2

envs = [Env1()]

for env in envs:

    timestamp = datetime.now()
    folder_name = (env.name() + timestamp.strftime("_%m_%d_%H_%M"))
    _dir = f"Output/{Path(__file__).stem}/{folder_name}"

    cycles = 10
    timeSteps = 500

    lr = 0.001
    while lr < 1:

        model = PPO('MlpPolicy', env, verbose=0, learning_rate=lr)

        print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

        for i in range(cycles):

            print("Beginning cycle " + str(i) + "of" + str(cycles))

            if not os.path.exists(_dir):
                os.makedirs(_dir)

            model.learn(total_timesteps=timeSteps)
            model.save(_dir)

        lr = lr + 0.005

    env.close()
