## Using reinforcement learning and 2D LiDAR observations to control a car and avoid obstacles in CARLA.

The purpose of this repository is to allow you to train an RL agent using stable-baselines3 algorithms to control a car 
and avoid obstacles using only 2D LiDAR observations in the CARLA simulator.

The environment is made up of 2 pieces which are only divided for readability:\
BackEnv.py inherits the base gymnasium environment and contains all the pieces for setting up the Carla environment. 
It also includes the functions for resetting and closing the environment. 
Unless you want to change which car and world you are using or what sensors are on the car this shouldn't need to be 
touched.\
Env.py inherits BackEnv.py and contains all the logic pertaining to: 1.Creating the observation in create_lidar_plane()
2.Calculating the action based on the agent's response in step() 3. Calculating the reward based on the agents state, 
also in step().

Trainer.py 
