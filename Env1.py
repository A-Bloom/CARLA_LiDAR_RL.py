# Action Space: 21 discrete throttle actions from 1 reverse to 1 forward 21 discrete steer actions from -1 to 1
# Observation Space:  Blank grid with ones signifying obstacles Precision of 1/4 meter
# Reward Function: Rewards for going faster up to 50 m/s with a discount for turning Max reward of 0.5

import math
from pathlib import Path
import numpy as np
import cv2 as cv
from BackEnv import BackEnv
from gymnasium import spaces
import carla


class CarEnv(BackEnv):

    def __init__(self):

        self.Show = True
        self.Verbose = True
        # self.Lidar_Depth = '30'
        self.Lidar_Resolution = 4  # Points per meter
        self.Lidar_PPS = '9000'  # Points/Second
        self.Lidar_RPS = '7'  # Rotations/Second
        self.delta_seconds = 0.05  # Lower number leads to higher precision/longer train time

        # host = '10.230.117.122'
        self.host = '127.0.0.1'
        self.port = 2000

        super(CarEnv, self).__init__()

        self.action_space = spaces.MultiDiscrete([21, 21])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.Lidar_Field, self.Lidar_Field),
                                            dtype=np.float32)

    def step(self, action):

        self.world.tick()

        done = False
        reward = 0

        self.step_counter += 1

        throttle = abs((action[0] - 10) / 10)
        steer = (action[1] - 10) / 10

        if action[0] >= 10:
            reverse = False
        else:
            reverse = True

        self.tesla.apply_control(carla.VehicleControl(steer=steer, reverse=reverse, throttle=throttle))

        if not reverse:

            velocity = self.tesla.get_velocity()
            abs_velocity = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
            if abs_velocity < 50:
                reward = abs_velocity / 100
            else:
                reward = 0.5 - abs_velocity / 100

        reward = reward - abs(steer) * reward

        if reward >= 1:
            self.init_location = self.tesla.get_location()

        if self.collision_sensed:
            reward = -1
            done = True

        if self.tesla.get_location().z < -20:
            done = True

        if self.Verbose and (self.step_counter % 500 == 0) and (self.step_counter != 0):
            print("-------------------------------------")
            print("Throttle action: " + str(action[0]) + "\nThrottle: " + str(throttle) +
                  "\nSteering Action: " + str(action[1]) + "\nSteering: " + str(steer) + "\nReward: " + str(reward))

        if self.Show:
            cv.imshow('Top View', self.camera_data)
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1] * 255)))
            cv.waitKey(1)

        print(self.Lidar_Depth)

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
