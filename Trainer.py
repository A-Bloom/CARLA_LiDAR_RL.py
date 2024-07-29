# TODO: Add callbacks to save the best policy and automatically delete earlier failed policies
#  to save memory and help with finding meaningful policies.
# TODO: Add callback to terminate an experiment when no progress is being made after n steps.
# TODO: Add support for algorithms in stable baselines3-contrib and SBX.
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common import utils
from datetime import datetime
from sys import platform
from subprocess import Popen
from pathlib import Path
from Env import Env
import zipfile
import json
import copy
import sys
import os


def train(experiment_runs=1, epochs=10, steps_per_epoch=1000, output_folder="Output", algorithms=['A2C'],
          connection_vars=None, debugging_vars=None, lidar_vars=None, reward_vars=None, action_vars=None,
          algorithm_vars=None, A2C_vars=None, DDPG_vars=None, DQN_vars=None, PPO_vars=None, SAC_vars=None,
          TD3_vars=None, restart=None):

    if restart is None:
        timestamp = datetime.now()
        folder_name = (output_folder + "/Experiment" + timestamp.strftime("_%m_%d_%H_%M"))
        log_dir = f"{folder_name}/TBLogs"
        os.makedirs(log_dir, exist_ok=True)
        # Saves all the experiment info to a file.
        experiment_info = open(f"{folder_name}/experiment_info.json", 'w')
        json.dump([experiment_runs, epochs, steps_per_epoch, algorithms,
                   connection_vars, debugging_vars, lidar_vars, reward_vars, action_vars, algorithm_vars,
                   A2C_vars, DDPG_vars, DQN_vars, PPO_vars, SAC_vars, TD3_vars], experiment_info)
        experiment_info.close()
        # Indices for all loops are initialized
        run_index, algorithm_index, experiment_index, configuration_index, epoch_index = 0, 0, 0, 0, 1
    else:
        # If the experiment exists already this determines where it halted and sets the indices.
        folder_name = restart
        log_dir = f"{folder_name}/TBLogs"
        run_info = open(f"{folder_name}/run_info.json")
        run_index, algorithm_index, experiment_index, configuration_index, epoch_index = json.load(run_info)
        run_info.close()

    if debugging_vars['Verbose']:
        verbose = 1
    else:
        verbose = 0

    device = "auto"
    if platform == "linux":
        os.popen(f"python -m tensorboard.main --logdir={log_dir}")
        print(f"Opening TensorBoard at {log_dir}")
        device = "cuda:1"  # !!Make sure to change this per your device's configuration!!
    elif platform == "win32":
        Popen(f"py -m tensorboard.main --logdir={log_dir}", creationflags=0x00000008)
        print(f"Opening TensorBoard at {log_dir}")

    # Gets all possible experiments.
    experiments = variableUnion(connection_vars, lidar_vars, reward_vars, action_vars, library=[])

    print("Stable Baselines3 running on " + str(utils.get_device(device='auto')))

    for run in range(run_index, experiment_runs):
        print(f"Beginning experiment run {run + 1} of {experiment_runs}")
        for algorithm in algorithms[algorithm_index:]:
            print(f"Running algorithm {algorithm}")
            # Gets all possible algorithm configurations
            algorithm_configurations = variableUnion(locals()[f"{algorithm}_vars"], algorithm_vars, library=[])
            experiments_len = len(experiments) * len(algorithm_configurations)
            for experiment in experiments[experiment_index:]:
                # Makes sure the experiment makes sense.
                if checkpoint(algorithm, experiment):
                    # Creates Environment.
                    env = Env(**experiment, **debugging_vars)
                    try:
                        for algorithm_configuration in algorithm_configurations[configuration_index:]:
                            print(f"Running experiment configuration {experiment_index * len(algorithm_configurations) + configuration_index + 1} of {experiments_len}")
                            # For CnnPolicy the LiDAR "images" are already normalized.
                            if algorithm_configuration['policy'] == 'CnnPolicy':
                                policy_kwargs = dict(normalize_images=False)
                            else:
                                policy_kwargs = None

                            if restart is None:
                                # Finds the algorithm from a string and creates a model.
                                model = globals()[algorithm](env=env, device=device, **algorithm_configuration,
                                                             tensorboard_log=log_dir, verbose=verbose,
                                                             policy_kwargs=policy_kwargs)
                                timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
                                tb_dir = f"{algorithm}_{timestamp}"
                                model_dir = f"{folder_name}/{algorithm}/{timestamp}"
                                os.makedirs(model_dir, exist_ok=True)
                                # Creates a file to log the parameters for this specific experiment.
                                var_info = open(f"{model_dir}/var_info.json", 'w')
                                json.dump([algorithm, experiment, algorithm_configuration], var_info)
                                var_info.close()
                            else:
                                # If the experiment is restarting this finds the last policy
                                # and uses it to build the model.
                                model_subdir = max(os.listdir(f"{folder_name}/{algorithm}"))
                                model_names = os.listdir(f"{folder_name}/{algorithm}/{model_subdir}")
                                model_names.sort()
                                model_path = f"{folder_name}/{algorithm}/{model_subdir}/{model_names[-2]}"
                                model = globals()[algorithm].load(model_path, env)

                                model_dir = f"{folder_name}/{algorithm}/{model_subdir}"
                                tb_dir = f"{algorithm}_{model_subdir}"
                                restart = None

                            for epoch in range(epoch_index, epochs + 1):
                                print(f"Beginning epoch {epoch} of {epochs}")
                                # Trains the model.
                                model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, tb_log_name=tb_dir)
                                # Saves the model at each epoch.
                                model.save(f"{model_dir}/{steps_per_epoch * epoch}")
                                # Puts the parameter log file in each .zip folder.
                                archive = zipfile.ZipFile(f"{model_dir}/{steps_per_epoch * epoch}.zip", 'a')
                                archive.write(f"{model_dir}/var_info.json", os.path.basename(f"{model_dir}/var_info.json"))
                                archive.close()
                                # Saves how far into the experiment it is.
                                run_info = open(f"{folder_name}/run_info.json", "w")
                                json.dump([run, algorithm_index, experiment_index, configuration_index, epoch], run_info)
                                run_info.close()
                            os.remove(f"{model_dir}/var_info.json")
                            epoch_index = 1
                            configuration_index += 1
                            del model
                    finally:
                        env.close()
                    configuration_index = 0
                experiment_index += 1
            experiment_index = 0
            algorithm_index += 1
        algorithm_index = 0
    os.remove(f"{folder_name}/run_info.json")


def checkpoint(algorithm, experiment):
    # Makes sure that the experiment makes sense or will run without errors.
    # If you find another nonsensical combination just stick it in.
    cleared = True
    discrete = ['A2C', 'DQN', 'PPO']
    continuous = ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3']
    try:
        if ((experiment['action_format'] == 'discrete' and algorithm not in discrete) or
                (experiment['action_format'] == 'continuous' and algorithm not in continuous) or
                (experiment['action_possibilities'] == 0 and experiment['constant_throttle'] == 0)):
            cleared = False
    except KeyError:
        pass

    return cleared


def variableUnion(*args, library):
    # TODO: Try to simplify this with itertools or linear algebra. It takes way too long.
    # This function takes all the possibilities in args and makes a library of all the possible combinations of
    # experiments. If you already have a library and want to add to it specify a library. Otherwise, specify library=[]
    # Don't ask me how it works. I wrote it, but I wouldn't be able to explain it. See TestBench4 to play with it.
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


# All of this is for calling the Trainer itself from commandline with the path to an existing experiment to restart it.
if __name__ == "__main__":
    # TODO: set this up with argparse with flags to change options like View and to allow the user to upgrade a
    #  single zip file with flags like steps and epochs.
    if len(sys.argv) == 2:
        if Path(sys.argv[1]).exists() and Path(f"{sys.argv[1]}/experiment_info.json").exists() and Path(
                f"{sys.argv[1]}/run_info.json").exists():

            experiment = open(f"{sys.argv[1]}/experiment_info.json")
            (experiment_runs, epochs, steps_per_epoch, algorithms, connection_options, debugging_options, lidar_options,
             reward_options, action_options, algorithm_options,
             A2C_options, DDPG_options, DQN_options, PPO_options, SAC_options, TD3_options) = json.load(experiment)
            experiment.close()

            train(connection_vars=connection_options, debugging_vars=debugging_options, lidar_vars=lidar_options,
                  reward_vars=reward_options, action_vars=action_options, algorithm_vars=algorithm_options,
                  experiment_runs=experiment_runs, epochs=epochs, steps_per_epoch=steps_per_epoch,
                  algorithms=algorithms, A2C_vars=A2C_options, DDPG_vars=DDPG_options, DQN_vars=DQN_options,
                  PPO_vars=PPO_options, SAC_vars=SAC_options, TD3_vars=TD3_options, restart=sys.argv[1])

        else:
            print("The File does not exist or the experiment was already completed. Exiting...")
    else:
        print("Invalid arguments. Exiting...")
