import Trainer
import numpy as np

# This script takes all possible combinations of environmental and learning hyperparameters and tests them.
# To just test one option, leave it as a single value. To test multiple, put it in an array.
# Comment it out to use default values.
# For examples see algorithms and algorithm_options section.
# This script will automatically ignore nonsensical possibilities like DDPG with a discrete action space

#  Carla Connection Options
connection_options = {
    'host': '127.0.0.1',  # localhost
    # 'host' ': '10.230.117.122',
    'port': 2000,
    'delta_seconds': 0.05  # Simulated seconds per frame.
    # Can also be an array for testing how much accuracy degrades with faster training time.
}

# Debugging Options
debugging_options = {
    'Show': True,
    'Verbose': False
}

# LiDAR/Observation Options
lidar_options = {
    'Lidar_Depth': '30',  # Furthest distance LiDAR reaches in meters. Must be string.
    'Lidar_Resolution': 1,  # Points generated per meter.
    'Lidar_PPS': '9000',  # Points/Second. Must be string.
    'Lidar_RPS': '7,',  # Rotations/Second. Must be string.
    'observation_format': 'image',  # 'points' or 'grid' or 'image'.
    # points is a list of (x,y) points from the car.
    # grid is an empty 2D array with ones representing obstacles.
    # image is grid with 2 more layers "np.dstack"ed on top to create an image
    # (normalized version of Lidar View).
    'Points_Per_Observation': 250  # Number of points put in the observation before resetting.
}

# Reward Options
# All the rewards are percentages of the reward and should add up to no more than 1 for normalization.
reward_options = {
    'reward_for_speed': 0,  # Closer to the speed limit. Normalized by speed_limit.
    'speed_limit': 50,  # In m/s. 14 m/s is ~31.3 mph.
    # If you don't want a speed limit chose a number that you think will be the cars max speed.

    'reward_for_displacement': 0.25,  # Further from the starting point. Normalized by displacement_reset.
    'displacement_reset': 200,

    'reward_for_destination': 0.5,
    # For being closer to the destination. Normalized by initial distance from destination.
    'destination_bonus': 1,  # Boosts the reward to 1 on arrival within 1 meter.

    'exponentialize_reward': 2,  # Will use this number as an exponent on the reward
    # to make it exponentially larger closer to the goal.
    'steps_b4_reset': 10000,  # Automatically resets after this many time steps.

    'min_speed': 1,  # in m/s
    'min_speed_punishment': -0.1,  # Will receive this if below min speed.
    'turn_punishment': 0.1,  # 0 to 1. Should increase the tendency to drive straight.
}

# Action Options
action_options = {
    'action_format': 'continuous',  # 'discrete' or 'continuous'
    'discrete_actions': 21,  # Only for discrete steer and or throttle, must be odd!
    'action_possibilities': 1,  # 0 for steer, 1 for throttle forward and steer,
    # 2 for throttle and steer, 3 for throttle, steer and break.
    'steer_cap': 1,  # 0 to 1. Doesn't allow the agent to steer harder than this number.
    'throttle_cap': 1,  # 0 to 1. Doesn't allow the agent to throttle harder than this number.
    'constant_throttle': 0.5,  # 0 to 1. For action_possibility 1.
    'turn_throttle_reduction': 0,  # 0 to 1. Reduces speed on turns for action_possibility 1.
}

# Run length variables
run_options = {
    'experiment_runs': 1,  # How many times to run the entire experiment.
    'epochs': 4,  # Saves the policy every epoch.
    'steps_per_epoch': 200,
}

# Reinforcement Learning Training Options. For more info see
# https://stable-baselines3.readthedocs.io/en/master/modules/base.html

# All options ['A2C', 'DDPG', 'DQN', 'PPO', 'SAC', 'TD3']
algorithms = ['SAC']  # This will test every other possibility with A2C and PPO

algorithm_options = {
    'policy': 'CnnPolicy',  # 'MlpPolicy', 'CnnPolicy', 'MultiInputPolicy'
    # Logarithmic range for learning rate to be tested with other possibilities.
    'learning_rate': [0.01, 0.1],
    # To test every option in a range you can use something like this.
    # (Must be a list because numpy arrays aren't json serializable)
    'gamma': 0.99
    # Comment out for default values.
    #'seed': None
}

# A2C specific options
A2C_options = {
    'n_steps': 1,
    'gae_lambda': 1.0,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'rms_prop_eps': 1e-05,
    'use_rms_prop': True,
    'use_sde': False,
    'sde_sample_freq': -1,
    'rollout_buffer_class': None,
    'rollout_buffer_kwargs': None,
    'normalize_advantage': False
}

# DDPG Specific options
DDPG_options = {
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'train_freq': 1,
    'gradient_steps': 1,
    'action_noise': None,
    'replay_buffer_class': None,
    'replay_buffer_kwargs': None,
    'optimize_memory_usage': False,
}

# DQN Specific options
DQN_options = {
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 32,
    'tau': 1.0,
    'train_freq': 4,
    'gradient_steps': 1,
    'replay_buffer_class': None,
    'replay_buffer_kwargs': None,
    'optimize_memory_usage': False,
    'target_update_interval': 10000,
    'exploration_fraction': 0.1,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    'max_grad_norm': 10
}

# PPO Specific options
PPO_options = {
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'clip_range_vf': None,
    'normalize_advantage': True,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'use_sde': False,
    'sde_sample_freq': -1,
    'rollout_buffer_class': None,
    'rollout_buffer_kwargs': None,
    'target_kl': None
}

# SAC Specific Options
SAC_options = {
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'train_freq': 1,
    'gradient_steps': 1,
    'action_noise': None,
    'replay_buffer_class': None,
    'replay_buffer_kwargs': None,
    'optimize_memory_usage': False,
    'ent_coef': 'auto',
    'target_update_interval': 1,
    'target_entropy': 'auto',
    'use_sde': False,
    'sde_sample_freq': -1,
    'use_sde_at_warmup': False
}

# TD3 Specific options
TD3_options = {
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'train_freq': 1,
    'gradient_steps': 1,
    'action_noise': None,
    'replay_buffer_class': None,
    'replay_buffer_kwargs': None,
    'optimize_memory_usage': False,
    'policy_delay': 2,
    'target_policy_noise': 0.2,
    'target_noise_clip': 0.5
}

Trainer.train(connection_vars=connection_options, debugging_vars=debugging_options, lidar_vars=lidar_options,
              reward_vars=reward_options, action_vars=action_options, algorithm_vars=algorithm_options, **run_options,
              algorithms=algorithms, A2C_vars=A2C_options, DDPG_vars=DDPG_options, DQN_vars=DQN_options,
              PPO_vars=PPO_options, SAC_vars=SAC_options, TD3_vars=TD3_options)
