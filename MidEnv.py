import math
import warnings
import numpy as np
import cv2 as cv
from pathlib import Path
from BackEnv import BackEnv
from gymnasium import spaces
import carla


class MidEnv(BackEnv):

    observation_format = 'grid'
    action_format = 'discrete'
    action_possibilities = 0
    discrete_actions = 21

    def __init__(self):

        super(MidEnv, self).__init__()

        if self.observation_format == 'points':
            self.observation = np.zeros((self.Points_Per_Observation, 2), dtype=np.float32)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.Points_Per_Observation, 2),
                                                dtype=np.float32)
        elif self.observation_space == 'grid':
            self.observation = np.zeros((self.Lidar_Field, self.Lidar_Field), dtype=np.float32)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.Lidar_Field, self.Lidar_Field),
                                                dtype=np.float32)
        else:
            warnings.warn("Invalid observation format! Exiting...")
            exit(-1)

        if self.action_format == 'discrete':
            if self.action_possibilities == 0:
                self.action_space = spaces.Discrete(self.discrete_actions)
            elif self.action_possibilities == 1:
                self.action_space = spaces.MultiDiscrete([self.discrete_actions, self.discrete_actions])
            elif self.action_possibilities == 2:
                self.action_space = spaces.MultiDiscrete([self.discrete_actions, self.discrete_actions, 2])
            else:
                warnings.warn("Invalid action possibility! Exiting...")
                exit(-1)
        elif self.action_format == 'continuous':
            if self.action_possibilities == 0:
                self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
            elif self.action_possibilities == 1:
                self.action_space = spaces.MultiDiscrete([self.discrete_actions, self.discrete_actions])
            elif self.action_possibilities == 2:
                self.action_space = spaces.MultiDiscrete([self.discrete_actions, self.discrete_actions, 2])
            else:
                warnings.warn("Invalid action possibility! Exiting...")
                exit(-1)



    def step(self, action):

        self.world.tick()

        self.done = False

        self.step_counter += 1

        steer = (action - 10) / 10

        throttle = 0.5 - 0.5*abs(steer)

        self.tesla.apply_control(carla.VehicleControl(steer=steer, reverse=False, throttle=throttle))

        self.reward = throttle

        if self.collision_sensed:
            self.reward = -1
            self.done = True

        if self.tesla.get_location().z < -20:
            self.done = True

        if self.Verbose and (self.step_counter % 500 == 0) and (self.step_counter != 0):
            print("-------------------------------------")
            print("Steering Action: " + str(action) + "\nSteering: " + str(steer) + "\nReward: " + str(self.reward))

        if self.Show:
            cv.imshow('Top View', self.camera_data)
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1] * 255)))
            cv.waitKey(1)

        return 0, 0, False, 0, {}

    def create_lidar_plane(self, points):

        raw = np.copy(np.frombuffer(points.raw_data, dtype=np.dtype('f4')))

        x_points = (((self.Lidar_Field - 1) / 2) + np.around(raw[1::4] * self.Lidar_Resolution)).astype(dtype=int)
        y_points = (((self.Lidar_Field - 1) / 2) - np.around(raw[::4] * self.Lidar_Resolution)).astype(dtype=int)

        self.lidar_data[0, y_points, x_points] = 1
        self.lidar_index += len(x_points)

        if self.lidar_index >= 250:
            self.lidar_data[1] = self.lidar_data[0]
            self.lidar_data[0] = np.zeros((self.Lidar_Field, self.Lidar_Field), dtype=np.dtype('f4'))
            self.lidar_index = 0

    def reset(self, **kwargs):
        super(MidEnv, self).reset()
        return 0, {}

    def close(self):
        super(MidEnv, self).close()

    def name(self):
        return Path(__file__).stem
