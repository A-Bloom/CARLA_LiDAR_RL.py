from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.evaluation import evaluate_policy
import os

base_path = "Output/Run2/Env3"

zips = []

dirs = os.listdir(base_path)

for sub_dir in dirs:
    files = os.listdir(f"{base_path}/{sub_dir}")
    for file in files:
        if file.endswith('.zip'):
            zips.append(f"{base_path}/{sub_dir}/{file}")
print(zips)

env = CarEnv()

print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

for path in zips:

    model = PPO.load(path, env)
    print(f"Evaluating {path}")
    print(evaluate_policy(model, env, 10))
    del model

env.close()
