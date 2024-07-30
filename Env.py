import math
import warnings
import numpy as np
import cv2 as cv
from BackEnv import BackEnv
from gymnasium import spaces
import carla
import time
import subprocess
import sys

reward = 0
done = False


class Env(BackEnv):

    def __init__(self,
                 Lidar_Depth='30',
                 Lidar_Resolution=4,
                 Lidar_PPS='9000',
                 Lidar_RPS='7',
                 host='127.0.0.1',
                 port=2000,
                 delta_seconds=0.05,
                 Verbose=False,
                 Show=True,
                 Manual=False,
                 observation_format='grid',
                 action_format='discrete',
                 action_possibilities=1,
                 discrete_actions=21,
                 steer_cap=1,
                 constant_throttle=0.5,
                 turn_throttle_reduction=0,
                 throttle_cap=1,
                 speed_limit=45,
                 reward_for_speed=0,
                 reward_for_displacement=0.25,
                 reward_for_destination=0.5,
                 displacement_reset=200,
                 Points_Per_Observation=0,
                 min_speed=1,
                 min_speed_punishment=-0.1,
                 exponentialize_reward=1,
                 steps_b4_reset=10000,
                 destination_bonus=1,
                 turn_punishment=0,
                 collision_course_punishment=0.5,
                 collision_course_range=3
                 ):


        if 'done' not in self.__dict__:
            self.collision_course_range = collision_course_range
            self.collision_course_punishment = collision_course_punishment
            self.distance_to_collision_object = self.collision_course_range
            self.turn_punishment = turn_punishment
            self.destination_bonus = destination_bonus
            self.steps_b4_reset = steps_b4_reset
            self.exponentialize_reward = exponentialize_reward
            self.min_speed_punishment = min_speed_punishment
            self.min_speed = min_speed
            self.Points_Per_Observation = Points_Per_Observation
            self.displacement_reset = displacement_reset
            self.reward_for_destination = reward_for_destination
            self.reward_for_displacement = reward_for_displacement
            self.reward_for_speed = reward_for_speed
            self.speed_limit = speed_limit
            self.throttle_cap = throttle_cap
            self.turn_throttle_reduction = turn_throttle_reduction
            self.constant_throttle = constant_throttle
            self.steer_cap = steer_cap
            self.discrete_actions = discrete_actions
            self.action_possibilities = action_possibilities
            self.action_format = action_format
            self.observation_format = observation_format
            self.Show = Show
            self.Verbose = Verbose
            self.delta_seconds = delta_seconds
            self.port = port
            self.host = host
            self.Lidar_RPS = Lidar_RPS
            self.Lidar_PPS = Lidar_PPS
            self.Lidar_Resolution = Lidar_Resolution
            self.Lidar_Depth = Lidar_Depth
            self.reward = 0
            self.done = False

        super(Env, self).__init__()

        self.lidar.listen(lambda data: self.create_lidar_plane(data))


        if self.Points_Per_Observation == 0:
            # This is the number of points I assume should be in one LiDAR revolution and is defaulted to.
            # TODO: From observation this seems to be inaccurate and a better value is 250. Figure out why.
            #  Uncomment the cv.imwrite() near the end of step() to get images of individual observations.
            self.Points_Per_Observation = round(int(self.Lidar_PPS) / int(self.Lidar_RPS))

        # Observation setup, see DefaultControlPanel for details.
        if self.observation_format == 'points':
            self.observation = np.zeros((self.Points_Per_Observation, 2), dtype=np.float32)
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.Points_Per_Observation, 2),
                                                dtype=np.float32)
            self.points = np.zeros((2, self.Points_Per_Observation, 2), dtype=np.float32)
        elif self.observation_format == 'grid':
            self.observation = np.zeros((self.Lidar_Field, self.Lidar_Field), dtype=np.float32)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.Lidar_Field, self.Lidar_Field),
                                                dtype=np.float32)
        elif self.observation_format == 'image':
            self.observation = np.zeros((3, self.Lidar_Field, self.Lidar_Field), dtype=np.uint8)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, self.Lidar_Field, self.Lidar_Field),
                                                dtype=np.uint8)
        else:
            warnings.warn("Invalid observation format! Exiting...")
            exit(-1)

        # Action setup, see DefaultControlPanel for details.
        if self.action_format == 'discrete':
            self.action_space = spaces.MultiDiscrete([self.discrete_actions, self.discrete_actions, 3])
        elif self.action_format == 'continuous':
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]),
                                           dtype=np.float32)
        else:
            warnings.warn("Invalid action format! Exiting...")
            exit(-1)

        # Needed to make the lowest discrete action -1 and the highest 1.
        self.discrete_augmentor = round((self.discrete_actions - 1) / 2)

        print("Initializing MidEnv")

    def step(self, action):

        try:
            # The client is in control, so you need to tick the server.
            self.world.tick()
        except RuntimeError:
            # TODO: Trying to reboot doesn't work. No clue why not. Try purposely crashing the server...
            print("Connection to Server Lost Attempting Reboot...")
            cv.destroyAllWindows()
            time.sleep(5)
            self.lidar.stop()
            self.camera.stop()
            self.collision.stop()
            if sys.platform == 'win32':
                subprocess.Popen("CarlaPath.bat")  # Edit this file to reflect your system specs
            else:  # linux or mac
                subprocess.call(['sh', './CarlaPath.sh'])  # Or edit this file to reflect your system specs
            time.sleep(20)
            self.__init__()
            print("Reboot Successful!")

            try:
                self.world.tick()
            except RuntimeError:
                print("Failed to Reconnect to Server, Shutting Down.")
                super(Env, self).close()

        self.done = False
        self.step_counter += 1
        brake = 0
        reverse = False



        # Action function, see DefaultControlPanel for details on action_format and action_possibilities.
        if self.action_format == 'discrete':
            steer = ((action[0] - self.discrete_augmentor) / self.discrete_augmentor) * self.steer_cap
            if self.action_possibilities == 0:
                throttle = self.constant_throttle * (1 - abs(steer) * self.turn_throttle_reduction)
            elif self.action_possibilities > 0:
                throttle = abs((action[1] - self.discrete_augmentor) / self.discrete_augmentor) * self.throttle_cap
                if action[1] < self.discrete_augmentor:
                    if self.action_possibilities == 1:
                        throttle = 0
                    elif self.action_possibilities > 1:
                        reverse = True
                if self.action_possibilities == 3:
                    brake = action[2] * 0.5


        elif self.action_format == 'continuous':
            steer = action[0] * self.steer_cap
            if self.action_possibilities == 0:
                throttle = self.constant_throttle * (1 - abs(steer) * self.turn_throttle_reduction)
            elif self.action_possibilities > 0:
                throttle = abs(action[1]) * self.throttle_cap
                if action[1] < 0:
                    if self.action_possibilities == 1:
                        throttle = 0
                    elif self.action_possibilities > 1:
                        reverse = True
                if self.action_possibilities == 3:
                    brake = action[2] * 1  # * 1 is to turn it into a float

        self.tesla.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse))

        # Reward function, as of right now it only gets rewarded for going straight
        if not reverse:

            velocity = self.tesla.get_velocity()
            abs_velocity = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
            velocity_reward = self.reward_for_speed * abs_velocity / self.speed_limit
            if abs_velocity > self.speed_limit:
                velocity_reward = self.reward_for_speed - velocity_reward

            displacement = math.sqrt((self.tesla.get_location().x - self.init_location.x) ** 2 +
                                     (self.tesla.get_location().y - self.init_location.y) ** 2)
            displacement_reward = self.reward_for_displacement * displacement / self.displacement_reset

            # Resets the displacement so that the agent has an impetus to keep moving.
            if displacement > self.displacement_reset:
                self.init_location = self.tesla.get_location()

            # Reward for getting closer to a destination.
            target = math.sqrt((self.tesla.get_location().x - self.target_location.x) ** 2 +
                               (self.tesla.get_location().y - self.target_location.y) ** 2)
            target_reward = self.reward_for_destination * (1 - target / self.distance_to_target)

            self.reward = ((velocity_reward + displacement_reward + target_reward) *
                           (1 - self.turn_punishment * abs(steer)) -
                           (1 - self.distance_to_collision_object/self.collision_course_range) *
                           self.collision_course_punishment)

            if self.reward < 0:
                self.reward = -(self.reward ** self.exponentialize_reward)
            else:
                self.reward = self.reward ** self.exponentialize_reward

        else:
            abs_velocity = self.min_speed
            displacement = 0
            target = self.distance_to_target


        # If it is going too slow it gets penalized.
        # You might want to consider self.reward -= self.min_speed_punishment as well.
        if abs_velocity < self.min_speed:
            self.reward = self.min_speed_punishment

        # As long as it is relatively close to the destination its doing fine.
        if target < 0.5:
            self.reward = self.destination_bonus

        if self.collision_sensed:
            self.reward = -1
            self.done = True

        # Sometimes the agent will randomly fall out of the planet. No kidding.
        if self.tesla.get_location().z < -20 or self.step_counter > self.steps_b4_reset:
            self.done = True

        if self.Verbose and (self.step_counter % 100 == 0) and (self.step_counter != 0):
            print("-------------------------------------")
            print(f"Steering Action: {action[0]} Steering: {steer}")
            print(f"Throttle Action: {action[1]} Throttle: {throttle}")
            print(f"Brake Action: {action[2]} Brake: {brake}")
            print(f"Velocity: {abs_velocity} Displacement: {displacement}")
            print(f"Distance to nearest forward object: {self.distance_to_collision_object}")
            print(f"Distance to target: {target} Reward: {self.reward}")
            # cv.imwrite(f"Output/Images/{time.time()}_lidar.jpg", np.dstack((self.blanks, self.blanks, self.lidar_data[1])) * 255)
            # cv.imwrite(f"Output/Images/{time.time()}_pic.jpg", self.camera_data)

        if self.Show:
            cv.imshow('Top View', self.camera_data)
            # OpenCV takes 3 channel BGR data, so I needed to stack two blank arrays on top of the
            # single channel lidar_data array *255 for color.
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1])) * 255)
            cv.waitKey(1)

        return self.observation, self.reward, False, self.done, {}

    def create_lidar_plane(self, points):

        raw = np.copy(np.frombuffer(points.raw_data, dtype=np.dtype('f4')))

        # Takes the x and y points from the raw data.
        # TODO: I know this works from observation but logic tells me that the 1 is in the wrong place.
        #  See TestBench2 for a simplified version of this.
        x_raw = raw[1::4]
        y_raw = raw[::4]
        length = len(x_raw)

        self.lidar_index += length

        if self.observation_format == 'grid' or self.observation_format == 'image' or self.Show:
            # To put the observation in a grid pattern the x,y points are formatted as integers in a 2D array.
            # See TestBench2 for a simplified version of this.
            x_points = (((self.Lidar_Field - 1) / 2) + np.around(x_raw * self.Lidar_Resolution)).astype(dtype=int)
            y_points = (((self.Lidar_Field - 1) / 2) - np.around(y_raw * self.Lidar_Resolution)).astype(dtype=int)

            # All of those indices are assigned 1
            self.lidar_data[0, y_points, x_points] = 1

            # If the observation buffer, lidar_data[0], fills, it gets replaced by lidar_data[1] and begins refilling.
            if self.lidar_index > self.Points_Per_Observation:
                self.lidar_data[1] = self.lidar_data[0]
                self.lidar_data[0] = np.zeros((self.Lidar_Field, self.Lidar_Field), dtype=np.uint8)

            if self.observation_format == 'grid':
                self.observation = self.lidar_data[1]
            elif self.observation_format == 'image':
                self.observation = np.stack((self.blanks, self.blanks, self.lidar_data[1]))

        # points[0] acts as a buffer until enough data points come from the LiDAR listener for a complete observation.
        # Then when self.lidar_index > self.Points_Per_Observation only enough points to fit enter the buffer
        # and points[1] gets swapped with points[0].
        # See TestBench3 for a simplified version of this.
        if self.observation_format == 'points' or self.collision_course_punishment > 0:
            if self.lidar_index > self.Points_Per_Observation:
                self.lidar_points[0][self.lidar_index - length:] = (
                        (np.dstack((x_raw, y_raw)).squeeze())[:(self.Points_Per_Observation - self.lidar_index)])
                self.lidar_points[1] = self.lidar_points[0]
                self.lidar_points[0] = np.zeros((self.Points_Per_Observation, 2), dtype=np.float32)
            else:
                self.lidar_points[0][(self.lidar_index - length):self.lidar_index] = np.dstack((x_raw, y_raw))

            # If punishment for collision course exists then find the closest point in the range.
            if self.collision_course_punishment > 0:
                self.distance_to_collision_object = self.collision_course_range
                for i in range(len(self.lidar_points[1])):
                    if (-1.5 <= self.lidar_points[1][i][0] <= 1.5) and (0 <= self.lidar_points[1][i][1] <= self.collision_course_range):
                        if math.hypot(self.lidar_points[1][i][0], self.lidar_points[1][i][1]) < self.distance_to_collision_object:
                            self.distance_to_collision_object = math.hypot(self.lidar_points[1][i][0], self.lidar_points[1][i][1])
            if self.observation_format == 'points':
                self.observation = self.lidar_points[1] / int(self.Lidar_Depth)

        if self.lidar_index > self.Points_Per_Observation:
            self.lidar_index = 0

    def reset(self, **kwargs):
        super(Env, self).reset()
        return self.observation, {}
