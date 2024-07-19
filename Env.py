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
                 destination_bonus=True,
                 turn_punishment=0
                 ):

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
            self.Points_Per_Observation = round(int(self.Lidar_PPS) / int(self.Lidar_RPS))

        # Observation setup
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
            self.observation = np.zeros((self.Lidar_Field, self.Lidar_Field, 3), dtype=np.float32)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.Lidar_Field, self.Lidar_Field, 3),
                                                dtype=np.float32)
        else:
            warnings.warn("Invalid observation format! Exiting...")
            exit(-1)

        # Action setup
        if self.action_format == 'discrete':
            self.action_space = spaces.MultiDiscrete([self.discrete_actions, self.discrete_actions, 3])
        elif self.action_format == 'continuous':
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]),
                                           dtype=np.float32)
        else:
            warnings.warn("Invalid action format! Exiting...")
            exit(-1)

        self.discrete_augmentor = round((self.discrete_actions - 1) / 2)

        print("Initializing MidEnv")

    def step(self, action):

        try:
            self.world.tick()
        except RuntimeError:
            # This doesn't work. No clue why not. Try purposely crashing the server...
            print("Connection to Server Lost Attempting Reboot...")
            cv.destroyAllWindows()
            time.sleep(5)
            if sys.platform == 'win32':
                subprocess.Popen("CarlaPath.bat")
            else:
                subprocess.call(['sh', './CarlaPath.sh'])
            time.sleep(20)
            print("Slept")
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

        # Action function
        if self.action_format == 'discrete':
            steer = ((action[0] - self.discrete_augmentor) / self.discrete_augmentor) * self.steer_cap
            if self.action_possibilities == 0:
                throttle = self.constant_throttle * (1 - abs(steer)) * (1 - self.turn_throttle_reduction)
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
                throttle = self.constant_throttle * (1 - abs(steer)) * (1 - self.turn_throttle_reduction)
            elif self.action_possibilities > 0:
                throttle = abs(action[1]) * self.throttle_cap
                if action[1] < 0:
                    if self.action_possibilities == 1:
                        throttle = 0
                    elif self.action_possibilities > 1:
                        reverse = True
                if self.action_possibilities == 3:
                    brake = action[2].astype(np.float64)

        self.tesla.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse))

        # Reward function
        if not reverse:

            velocity = self.tesla.get_velocity()
            abs_velocity = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
            velocity_reward = self.reward_for_speed * abs_velocity / self.speed_limit
            if abs_velocity > self.speed_limit:
                velocity_reward = self.reward_for_speed - velocity_reward

            displacement = math.sqrt((self.tesla.get_location().x - self.init_location.x) ** 2 +
                                     (self.tesla.get_location().y - self.init_location.y) ** 2)
            displacement_reward = self.reward_for_displacement * displacement / self.displacement_reset

            if displacement > self.displacement_reset:
                self.init_location = self.tesla.get_location()

            target = math.sqrt((self.tesla.get_location().x - self.target_location.x) ** 2 +
                               (self.tesla.get_location().y - self.target_location.y) ** 2)
            target_reward = self.reward_for_destination * (1 - target / self.distance_to_target)

            self.reward = ((velocity_reward + displacement_reward + target_reward)*(1-self.turn_punishment)) ** self.exponentialize_reward
        else:
            abs_velocity = 0
            displacement = 0
            target = self.distance_to_target

        if abs_velocity < self.min_speed:
            self.reward = self.min_speed_punishment

        if target < 0.5 and self.destination_bonus:
            self.reward = 1
            self.done = True

        if self.collision_sensed:
            self.reward = -1
            self.done = True

        if self.tesla.get_location().z < -20 or self.step_counter > self.steps_b4_reset:
            self.done = True

        if self.Verbose and (self.step_counter % 100 == 0) and (self.step_counter != 0):
            print("-------------------------------------")
            print(f"Steering Action: {action[0]} Steering: {steer}")
            print(f"Throttle Action: {action[1]} Throttle: {throttle}")
            print(f"Brake Action: {action[2]} Brake: {brake}")
            print(f"Velocity: {abs_velocity} Displacement: {displacement}")
            print(f"Distance to target: {target} Reward: {self.reward}")
            # cv.imwrite(f"Output/Images/{time.time()}.jpg", np.dstack( (self.blanks, self.blanks, self.lidar_data[1] * 255)))

        if self.Show:
            cv.imshow('Top View', self.camera_data)
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1])) * 255)
            cv.waitKey(1)

        return self.observation, self.reward, False, self.done, {}

    def create_lidar_plane(self, points):

        raw = np.copy(np.frombuffer(points.raw_data, dtype=np.dtype('f4')))

        x_raw = raw[1::4]
        y_raw = raw[::4]
        length = len(x_raw)

        self.lidar_index += length

        if self.observation_format == 'grid' or self.observation_format == 'image' or self.Show:
            x_points = (((self.Lidar_Field - 1) / 2) + np.around(x_raw * self.Lidar_Resolution)).astype(dtype=int)
            y_points = (((self.Lidar_Field - 1) / 2) - np.around(y_raw * self.Lidar_Resolution)).astype(dtype=int)

            self.lidar_data[0, y_points, x_points] = 1

            if self.lidar_index > self.Points_Per_Observation:
                self.lidar_data[1] = self.lidar_data[0]
                self.lidar_data[0] = np.zeros((self.Lidar_Field, self.Lidar_Field), dtype=np.dtype('f4'))

            if self.observation_format == 'grid':
                self.observation = self.lidar_data[1]
            elif self.observation_format == 'image':
                self.observation = np.dstack((self.blanks, self.blanks, self.lidar_data[1]))

        if self.observation_format == 'points':
            if self.lidar_index > self.Points_Per_Observation:
                self.points[0][self.lidar_index - length:] = ((np.dstack((x_raw, y_raw)).squeeze())[
                                                              :(self.Points_Per_Observation - self.lidar_index)]) / int(
                    self.Lidar_Depth)
                self.points[1] = self.points[0]
                self.points[0] = np.zeros((self.Points_Per_Observation, 2), dtype=np.float32)
            else:
                self.points[0][(self.lidar_index - length):self.lidar_index] = np.dstack((x_raw, y_raw)) / int(
                    self.Lidar_Depth)

            self.observation = self.points[1]

        if self.lidar_index > self.Points_Per_Observation:
            self.lidar_index = 0

    def reset(self, **kwargs):
        super(Env, self).reset()
        return self.observation, {}
