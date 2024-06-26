# Action Space: 21 discrete steer actions from -1 to 1
# Observation Space:  Blank grid with ones signifying obstacles Precision of 1/4 meter
# Reward Function: Reward for throttle, throttle is decreased proportionally to steer

import math
import numpy as np
import cv2 as cv
from pathlib import Path
from BackEnv import BackEnv
from gymnasium import spaces
import carla


class CarEnv(BackEnv):

    def __init__(self):

        self.Show = True
        self.Verbose = True
        self.Lidar_Depth = '30'
        self.Lidar_Resolution = 4  # Points per meter
        self.Lidar_PPS = '9000'  # Points/Second
        self.Lidar_RPS = '7'  # Rotations/Second
        self.delta_seconds = 0.05  # Lower number leads to higher precision/longer train time

        # host = '10.230.117.122'
        self.host = '127.0.0.1'
        self.port = 2000

        super(CarEnv, self).__init__()

        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.Lidar_Field, self.Lidar_Field),
                                            dtype=np.float32)

    def step(self, action):

        self.world.tick()

        done = False

        self.step_counter += 1

        steer = (action - 10) / 10

        throttle = 0.5 - 0.5*abs(steer)

        self.tesla.apply_control(carla.VehicleControl(steer=steer, reverse=False, throttle=throttle))

        reward = throttle

        if self.collision_sensed:
            reward = -1
            done = True

        if self.tesla.get_location().z < -20:
            done = True

        if self.Verbose and (self.step_counter % 500 == 0) and (self.step_counter != 0):
            print("-------------------------------------")
            print("Steering Action: " + str(action) + "\nSteering: " + str(steer) + "\nReward: " + str(reward))

        if self.Show:
            cv.imshow('Top View', self.camera_data)
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1] * 255)))
            cv.waitKey(1)

        return self.lidar_data[1], reward, False, done, {}

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
        super(CarEnv, self).reset()
        return self.lidar_data[1], {}

    def close(self):
        super(CarEnv, self).close()

    def name(self):
        return Path(__file__).stem
