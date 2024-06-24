from stable_baselines3 import PPO
from stable_baselines3.common import utils
import os
from Env import CarEnv
import time

env = CarEnv()

model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.001)

timeSteps = 50000

# check_env(env)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

model.learn(total_timesteps=timeSteps, reset_num_timesteps=False)

models_dir = f"Output/models/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model.save(f"{models_dir}/{timeSteps}")

env.close()
