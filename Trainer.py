from stable_baselines3 import PPO
from stable_baselines3.common import utils
import os
from Env1 import CarEnv
import time

env = CarEnv()

model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.001)

cycles = 10
timeSteps = 500

#  check_env(env)
print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))
print("Beginning cycle 1")

model.learn(total_timesteps=timeSteps)

models_dir = f"Output/models/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model.save(f"{models_dir}/{timeSteps}")

for i in range(cycles-1):
    print("Beginning cycle " + str(i+2))
    model.load(f"{models_dir}/{timeSteps}")
    model.learn(total_timesteps=timeSteps)
    model.save(f"{models_dir}/{timeSteps}")

env.close()
