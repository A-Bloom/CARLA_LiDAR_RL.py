## Using reinforcement learning and 2D LiDAR observations to control a car and avoid obstacles in Carla.

The purpose of this repository is to allow you to train an RL agent using stable-baselines3 algorithms to control a car 
and avoid obstacles using only 2D LiDAR observations in the CARLA simulator. 
It is built in a way to let you design complex experiments and have the code run all of them in sequence 
and log everything for you.

Everything was built using Carla 0.9.15 (https://github.com/carla-simulator/carla/releases/tag/0.9.15/) and Python 3.10.
Other dependencies are listed in requirements.txt and can be installed with: `pip install -r requirements.txt`

Launch the Carla server before running anything else.

ControlPanel.py is where you set up your experiment. If you only have a single possibility you want to test,
leave the value as a single value. If you want to test multiple possibilities, put two values in an array,
and it will test them both in sequence. You can array any number of parameters, and it will test all the combinations.
Run ControlPanel.py to start the experiment.

Alternatively, to continue a terminated experiment, you can run Trainer.py from command line with an argument specifying
the path to the terminated experiment. Feasibly this can be used to change environmental hyperparameters mid-training 
like action_possibilities, but be warned that changing others like LidarDepth will break the trainer.

Trainer.py calculates, runs and logs all the experiments.

The environment is made up of 2 pieces which are only divided for readability:\
BackEnv.py inherits the base gymnasium class and contains all the pieces for setting up the Carla environment. 
It also includes the functions for resetting and closing the environment. 
Unless you want to change which car and world you are using or what sensors are on the car this shouldn't need to be 
touched.\
Env.py inherits BackEnv.py and contains all the logic pertaining to: 1.Creating the observation in create_lidar_plane()
2.Calculating the action based on the agent's response in step() 3. Calculating the reward based on the agents state, 
also in step().

DefaultControlPanel.py is just a reference for all the default values.

Evaluator.py allows you to visualize policies. You can evaluate any number of policies in sequence.
Either edit the default file path or run Evaluator.py in command line and add paths to any number of files and folders
you want to evaluate.

CleanUp.py just clears all the actors from the world and unlocks the world from synchronous mode.
Very useful if you have a couple of failed runs and there are dead cars scattered all over the place.

generate_traffic.py is written by the Carla devs. 
Launch it before launching ControlPanel.py for a more exciting environment.

The CarlaPath files are files to allow Carla to be re-launched by the client if it crashes. 
It should be modified per your system. It currently doesn't work properly. Try crashing the server on purpose...

ManualControl.py does exactly what it sounds like it should do.

The TestBenches are simplified versions of some of the more complex pieces of code. 
They are referenced throughout the comments.

This script has in no means been tested in all configurations, use at your own risk.
