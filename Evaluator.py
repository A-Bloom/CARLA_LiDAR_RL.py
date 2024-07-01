from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from Env3 import CarEnv

env = CarEnv()

path = "Output/Models/Trainer/Env3_07_01_12_02"

model = PPO.load(path, env)

# check_env(env)

print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

print(evaluate_policy(model, env, 10))

env.close()
