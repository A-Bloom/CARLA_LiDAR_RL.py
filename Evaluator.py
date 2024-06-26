from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from Env1 import CarEnv

env = CarEnv()

path = "Models/500"

model = PPO.load(path, env)

# check_env(env)

print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

print(evaluate_policy(model, env, 1))

env.close()
