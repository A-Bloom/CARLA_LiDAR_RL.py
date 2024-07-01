import math
import warnings
import numpy as np
import cv2 as cv
from pathlib import Path
from BackEnv import BackEnv
from gymnasium import spaces
import carla


class MidEnv(BackEnv):

    View = True
    observation_format = 'grid'
    action_format = 'discrete'
    action_possibilities = 0
    discrete_actions = 21
    steer_cap = 1
    constant_throttle = 0.5
    turn_throttle_reduction = 0
    throttle_cap = 1
    reward = 0
    done = False
    speed_limit = 45
    reward_distribution = [0.5, 0.5]
    displacement_reset = 200
    Points_Per_Observation = 0

    def __init__(self):

        super(MidEnv, self).__init__()

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
        else:
            warnings.warn("Invalid observation format! Exiting...")
            exit(-1)

        # Action setup
        if self.action_format == 'discrete':
            self.action_space = spaces.MultiDiscrete([self.discrete_actions, self.discrete_actions, 3])
        elif self.action_format == 'continuous':
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        else:
            warnings.warn("Invalid action format! Exiting...")
            exit(-1)

        self.discrete_augmentor = round((self.discrete_actions - 1)/2)

        print("Initializing MidEnv")

    def step(self, action):

        self.world.tick()
        self.done = False
        self.step_counter += 1
        brake = 0
        reverse = False

        #Action function
        if self.action_format == 'discrete':
            steer = ((action[0] - self.discrete_augmentor) / self.discrete_augmentor)*self.steer_cap
            if self.action_possibilities == 0:
                throttle = self.constant_throttle * (1 - abs(steer)) * (1 - self.turn_throttle_reduction)
            elif self.action_possibilities > 0:
                throttle = abs((action[1] - self.discrete_augmentor) / self.discrete_augmentor)*self.throttle_cap
                if action[1] < self.discrete_augmentor:
                    if self.action_possibilities == 1:
                        throttle = 0
                    elif self.action_possibilities > 1:
                        reverse = True
                if self.action_possibilities == 3:
                    brake = action[2]*0.5

        elif self.action_format == 'continuous':
            steer = action[0]*self.steer_cap
            if self.action_possibilities == 0:
                throttle = self.constant_throttle * (1 - abs(steer)) * (1 - self.turn_throttle_reduction)
            elif self.action_possibilities > 0:
                throttle = abs(action[1])*self.throttle_cap
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
            speed = self.reward_distribution[0] * (abs_velocity / self.speed_limit)
            if abs_velocity > self.speed_limit:
                speed = self.reward_distribution[0]-speed

            displacement = self.reward_distribution[1]*(math.sqrt((self.tesla.get_location().x - self.init_location.x) ** 2 +
                                (self.tesla.get_location().y - self.init_location.y) ** 2) / self.displacement_reset)
            if displacement > self.reward_distribution[1]:
                self.init_location = self.tesla.get_location()

            self.reward = speed+displacement

        if self.collision_sensed:
            self.reward = -1
            self.done = True

        if self.tesla.get_location().z < -20:
            self.done = True

        if self.Verbose and (self.step_counter % 500 == 0) and (self.step_counter != 0):
            print("-------------------------------------")
            print("Steering Action: " + str(action[0]) + " Steering: " + str(steer))
            print("Throttle Action: " + str(action[1]) + " Throttle: " + str(throttle))
            print("Brake Action: " + str(action[2]) + " Brake: " + str(brake))
            print("Reward: " + str(self.reward))

        if self.Show:
            cv.imshow('Top View', self.camera_data)
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1] * 255)))
            cv.waitKey(1)

        return self.observation, self.reward, False, self.done, {}

    def create_lidar_plane(self, points):

        raw = np.copy(np.frombuffer(points.raw_data, dtype=np.dtype('f4')))

        x_raw = raw[1::4]
        y_raw = raw[::4]
        length = len(x_raw)

        self.lidar_index += length

        if self.observation_format == 'grid' or self.View:
            x_points = (((self.Lidar_Field - 1) / 2) + np.around(x_raw * self.Lidar_Resolution)).astype(dtype=int)
            y_points = (((self.Lidar_Field - 1) / 2) - np.around(y_raw * self.Lidar_Resolution)).astype(dtype=int)

            self.lidar_data[0, y_points, x_points] = 1

            if self.lidar_index >= self.Points_Per_Observation:
                self.lidar_data[1] = self.lidar_data[0]
                self.lidar_data[0] = np.zeros((self.Lidar_Field, self.Lidar_Field), dtype=np.dtype('f4'))
                # self.lidar_index = 0

            if self.observation_format == 'grid':
                self.observation = self.lidar_data[1]

        if self.observation_format == 'points':
            if self.lidar_index > self.Points_Per_Observation:
                self.points[0][self.lidar_index-length:] = ((np.dstack((x_raw, y_raw)).squeeze())[:(self.Points_Per_Observation-self.lidar_index)])/int(self.Lidar_Depth)
                self.points[1] = self.points[0]
                self.points[0] = np.zeros((self.Points_Per_Observation, 2), dtype=np.float32)
                self.lidar_index = 0
            else:
                self.points[0][(self.lidar_index - length):self.lidar_index] = np.dstack((x_raw, y_raw))/int(self.Lidar_Depth)

            self.observation = self.points[1]


    def reset(self, **kwargs):
        super(MidEnv, self).reset()
        return self.observation, {}

    def name(self):
        return Path(__file__).stem
