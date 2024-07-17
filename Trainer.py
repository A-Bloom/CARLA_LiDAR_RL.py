
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common import utils
from pathlib import Path
from datetime import datetime
from sys import platform
from subprocess import Popen
from MidEnv import MidEnv
import copy
import os


def train(A2C_vars=None, DDPG_vars=None, DQN_vars=None, PPO_vars=None, SAC_vars=None, TD3_vars=None,
          connection_vars=None, debugging_vars=None, lidar_vars=None, reward_vars=None, algorithm_vars=None,
          experiment_runs=1, epochs=10, steps_per_epoch=1000, algorithms=None):

    timestamp = datetime.now()
    folder_name = ("Output/" + Path(__file__).stem + timestamp.strftime("_%m_%d_%H_%M"))
    log_dir = f"{folder_name}/TBLogs"
    os.makedirs(log_dir, exist_ok=True)
    print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

    device = "auto"
    if platform == "linux":
        os.popen(f"python -m tensorboard.main --logdir={log_dir}")
        print(f"Opening TensorBoard at {log_dir}")
        device = "cuda:1"
    elif platform == "win32":
        Popen(f"py -m tensorboard.main --logdir={log_dir}", creationflags=0x00000008)
        print(f"Opening TensorBoard at {log_dir}")

    experiments = variableUnion(connection_vars, debugging_vars, lidar_vars, reward_vars)

    for run in range(experiment_runs):
        for algorithm in algorithms:
            algorithm_factors = variableUnion(locals()[f"{algorithm}_vars"], algorithm_vars)
            for experiment in experiments:
                env = MidEnv(**experiment)
                for algorithm_factor in algorithm_factors:
                    model = locals()[algorithm](env, **algorithm_factor)
                    for epoch in range(epochs):
                        model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, tb_log_name=tb_dir)
                        model.save()



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
            model.learn(total_timesteps=timeSteps,  reset_num_timesteps=False, tb_log_name=tb_dir)
            os.makedirs(model_dir, exist_ok=True)
            model.save(f"{model_dir}/{i*timeSteps}")

        lr = lr + 0.03
        del model

    env.close()


    def variableUnion(*args, library=[]):
        interim = []

        for variable in args:
            if type(variable) is dict:
                for key, value in variable.items():
                    if type(value) is list:
                        if len(library) == 0:
                            for index in value:
                                interim.append({key: index})
                        else:
                            for index in value:
                                for i in range(len(library)):
                                    library[i].update({key: index})
                                interim.extend(copy.deepcopy(library))

                        library = interim
                        interim = []
                    else:
                        offset = 0
                        if len(library) == 0:
                            library.append({key: value})
                            offset = 1
                        for final in range(offset, len(library)):
                            library[final].update({key: value})

        return library
