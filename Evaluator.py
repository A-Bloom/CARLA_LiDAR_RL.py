from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common import utils
from stable_baselines3.common.evaluation import evaluate_policy
import zipfile
import json
import sys
import os
from pathlib import Path
from Env import Env

path = r"C:\Users\abche\Documents\F1_10_Mini_Autonomous_Driving\LIDAR_1\Output\Experiment_07_19_09_09\A2C\07_19_09_09_41\500.zip"
debugging_options = {'Show': True, 'Verbose': True}
episodes_to_evaluate = 10


def evaluate(path_to_zip, debugging_vars, n_eval_episodes):
    try:
        archive = zipfile.ZipFile(path_to_zip, 'r')
        var_info = archive.open('var_info.json')
        algorithm, experiment, algorithm_configuration = json.load(var_info)
        env = Env(**experiment, **debugging_vars)
        model = globals()[algorithm].load(path_to_zip, env)
        print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))
        # For this problem deterministic needs to be false because sometimes the agent gets stuck repeating a throttle
        # of -1 and needs some stochasticity to get it out of that loop.
        stats = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=False)
        print( "  ________________________________________________________________________ ")
        print(f" | Mean Reward: {stats[0]} Standard Deviation: {stats[1]} |")
        print( "  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ")
    finally:
        env.close()


if len(sys.argv) == 1:
    try:
        print(f"Evaluating {path}")
        evaluate(path, debugging_options, episodes_to_evaluate)
    except:
        print(f"{path} could not be evaluated. Unknown Error.")
else:
    for arg in sys.argv[1:]:
        location = Path(arg)
        if location.exists():
            if location.is_file():
                if arg.endswith('.zip'):
                    try:
                        print(f"Evaluating {arg}")
                        evaluate(arg, debugging_options, episodes_to_evaluate)
                    except:
                        print(f"{arg} could not be evaluated. Unknown Error.")
                else:
                    print(f"{arg} is not a zip file.")
            elif location.is_dir:
                files = os.listdir(arg)
                for file in files:
                    if file.endswith('.zip'):
                        try:
                            print(f"Evaluating {arg}")
                            evaluate(arg, debugging_options, episodes_to_evaluate)
                        except:
                            print(f"{arg} could not be evaluated. Unknown Error.")
        else:
            print(f"{arg} does not exist.")

