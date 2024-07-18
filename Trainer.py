from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common import utils
from datetime import datetime
from sys import platform
from subprocess import Popen
from MidEnv import MidEnv
import zipfile
import json
import copy
import os


def train(experiment_runs=1, epochs=10, steps_per_epoch=1000, algorithms=['A2C'],
          connection_vars=None, debugging_vars=None, lidar_vars=None, reward_vars=None, action_vars=None, algorithm_vars=None,
          A2C_vars=None, DDPG_vars=None, DQN_vars=None, PPO_vars=None, SAC_vars=None, TD3_vars=None):

    timestamp = datetime.now()
    folder_name = ("Output/Experiment" + timestamp.strftime("_%m_%d_%H_%M"))
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

    experiment_info = open(f"{folder_name}/experiment_info.json", 'w')
    json.dump([experiment_runs, epochs, steps_per_epoch, algorithms,
               connection_vars, debugging_vars, lidar_vars, reward_vars, action_vars, algorithm_vars,
               A2C_vars, DDPG_vars, DQN_vars, PPO_vars, SAC_vars, TD3_vars], experiment_info)
    experiment_info.close()

    experiments = variableUnion(connection_vars, lidar_vars, reward_vars, action_vars, library=[])

    for run in range(experiment_runs):
        print(f"Beginning run {run+1} of {experiment_runs}")
        for algorithm in algorithms:
            print(f"Running algorithm {algorithm}")
            algorithm_factors = variableUnion(locals()[f"{algorithm}_vars"], algorithm_vars, library=[])
            for experiment in experiments:
                env = MidEnv(**experiment, **debugging_vars)
                for algorithm_factor in algorithm_factors:
                    print("here")
                    model = globals()[algorithm](env, device, **algorithm_factor)
                    timestamp = datetime.now().strftime("_%m_%d_%H_%M")
                    tb_dir = f"{algorithm}_{timestamp}"
                    model_dir = f"{folder_name}/{algorithm}/{timestamp}"
                    for epoch in range(epochs):
                        print(f"Beginning epoch {epoch+1} of {epochs}")
                        model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, tb_log_name=tb_dir)
                        os.makedirs(model_dir, exist_ok=True)
                        model.save(f"{model_dir}/{steps_per_epoch * epoch}")
                        var_info = open(f"{model_dir}/var_info.json", 'w')
                        json.dump([algorithm, experiment, algorithm_factor], var_info)
                        var_info.close()
                        archive = zipfile.ZipFile(f"{model_dir}/{steps_per_epoch * epoch}", 'a')
                        archive.write(f"{model_dir}/var_info.json", os.path.basename(f"{model_dir}/var_info.json"))
                        archive.close()
                        os.remove(f"{model_dir}/var_info.json")
                    del model
                env.close()


def variableUnion(*args, library):
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
