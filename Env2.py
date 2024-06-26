import math
import numpy as np
import cv2 as cv
import gymnasium as gym
from gymnasium import spaces
import carla
import BackEnv


class CarEnv(gym.Env):

    def __init__(self):

        super(CarEnv, self).__init__()

        self.Show = True
        self.Verbose = True
        self.Lidar_Depth = '30'
        self.Lidar_Resolution = 4  # Points per meter
        self.Lidar_PPS = '9000'  # Points/Second
        self.Lidar_RPS = '7'  # Rotations/Second

        # host = '10.230.117.122'
        self.host = '127.0.0.1'
        self.port = 2000

        BackEnv.setup(self)

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
            # Velocity in m/s; 14 m/s is about 31.3 mph
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
            cv.imshow('Top View', self.camera_data['image'])
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1] * 255)))

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
        BackEnv.reset(self)

        return self.lidar_data[1], {}

    def collided(self, data):
        self.collision_sensed = True

    def create_image(self, pic):
        if self.Show:self.camera_data['image'] = np.reshape(np.copy(pic.raw_data), (pic.height, pic.width, 4))

    def close(self):
        BackEnv.close(self)
