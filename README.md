## Using reinforcement learning and 2D LiDAR observations to control a car and avoid obstacles in Carla.

### Overview:

The purpose of this repository is to allow you to train an RL agent using stable-baselines3 algorithms to control a car 
and avoid obstacles using only 2D LiDAR observations in the CARLA simulator. 
It is built in a way to let you design complex experiments and have the code run all of them in sequence 
and log everything for you.

### Setup:

Download the [Carla 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15/) server.

For Windows, you might need legacy DirectX drivers to run Carla found 
[here.](https://www.microsoft.com/en-gb/download/details.aspx?id=35&irgwc=1&OCID=AIDcmm549zy227_aff_7815_119570&tduid=%28ir__wf1t6jfdiwkfdzevufswalhllm2xdqxerm0icgxq00%29%287815%29%28119570%29%285728363%29%28lwqs5u6ave03es170tipy%29&irclickid=_wf1t6jfdiwkfdzevufswalhllm2xdqxerm0icgxq00)

Everything was built using [Python 3.10](https://www.python.org/downloads/release/python-31011/) 
but other versions may work as well.

Other dependencies are listed in requirements.txt and can be installed with: `pip install -r requirements.txt`

Launch the Carla server before running anything else.

Set up ControlPanel.py with the parameters you want and run it directly.

### The Code:

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
This is usually run by the environment but can also be run directly if during testing you have a couple of failed runs
and there are dead cars scattered all over the place.

generate_traffic.py is written by the Carla devs. 
Launch it before launching ControlPanel.py for a more exciting environment.

The CarlaPath files are files to allow Carla to be re-launched by the client if it crashes. 
It should be modified per your system. It currently doesn't work properly. Try crashing the server on purpose...

ManualControl.py does exactly what it sounds like it should do.

The TestBenches are simplified versions of some of the more complex pieces of code. 
They are referenced throughout the comments.

### Disclaimer:

This script has in no means been tested in all configurations, use at your own risk.
