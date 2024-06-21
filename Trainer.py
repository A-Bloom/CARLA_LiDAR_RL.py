from stable_baselines3 import PPO
import os
from Env import CarEnv
import time

env = CarEnv()

model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.001)

timeSteps = 50000

# check_env(env)

model.learn(total_timesteps=timeSteps, reset_num_timesteps=False)

models_dir = f"Output/models/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model.save(f"{models_dir}/{timeSteps}")

env.close()
