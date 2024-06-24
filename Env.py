import glob
import os
import sys
import random
import math
import time

import numpy as np
import cv2 as cv
import gymnasium as gym
from gymnasium import spaces

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

Show = True
Verbose = True
ExtraVerbose = False
Lidar_Depth = '30'
Lidar_Resolution = 4  # Points per meter
Lidar_Field = int(Lidar_Depth) * 2 * Lidar_Resolution + 1
Lidar_PPS = '9000'  # Points/Second
Lidar_RPS = '7'  # Rotations/Second
Points_Per_Observation = int(int(Lidar_PPS) / int(Lidar_RPS))

# host = '10.230.117.122'
host = '127.0.0.1'
port = 2000

if Show:
    cv.namedWindow('Top View', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Lidar View', cv.WINDOW_AUTOSIZE)


class CarEnv(gym.Env):

    def __init__(self):
        super(CarEnv, self).__init__()

        self.action_space = spaces.MultiDiscrete([21, 21])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(Lidar_Field, Lidar_Field), dtype=np.float32)

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        # self.world = self.client.load_world(random.choice(self.client.get_available_maps()))

        self.spectator = self.world.get_spectator()

        if self.spectator.get_transform().location.z < 150:
            self.spectator.set_transform(carla.Transform(carla.Location(x=50, y=50, z=250), carla.Rotation(pitch=-90)))

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.number_of_spawn_points = len(self.spawn_points)

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.01
        self.world.apply_settings(self.settings)

        self.bps = self.world.get_blueprint_library()
        self.tesla_bp = self.bps.find('vehicle.tesla.model3')
        self.lidar_bp = self.bps.find('sensor.lidar.ray_cast')
        self.camera_bp = self.bps.find('sensor.camera.rgb')
        self.collision_bp = self.bps.find('sensor.other.collision')

        self.tesla_bp.set_attribute('color', '255, 110, 199')
        while True:
            try:
                self.tesla = self.world.spawn_actor(self.tesla_bp, random.choice(self.spawn_points))
                break
            except:
                print("Retrying Spawn")

        self.abs_throttle = 0
        self.step_counter = 0
        self.init_location = self.tesla.get_location()

        self.lidar_init_transform = carla.Transform(carla.Location(z=2))
        self.lidar_bp.set_attribute('channels', '1')
        self.lidar_bp.set_attribute('range', Lidar_Depth)
        self.lidar_bp.set_attribute('rotation_frequency', Lidar_RPS)
        self.lidar_bp.set_attribute('upper_fov', '0')
        self.lidar_bp.set_attribute('lower_fov', '0')
        self.lidar_bp.set_attribute('points_per_second', Lidar_PPS)
        self.lidar = self.world.spawn_actor(self.lidar_bp, self.lidar_init_transform, attach_to=self.tesla)
        self.lidar_data = np.zeros((2, Lidar_Field, Lidar_Field))
        self.blanks = np.zeros((Lidar_Field, Lidar_Field))
        self.lidar_index = 0
        self.lidar.listen(lambda data: self.create_lidar_plane(data))

        self.camera_init_transform = carla.Transform(carla.Location(z=30), carla.Rotation(pitch=-90))
        self.camera_bp.set_attribute('image_size_x', '768')
        self.camera_bp.set_attribute('image_size_y', '768')
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_init_transform, attach_to=self.tesla)
        self.image_w = self.camera_bp.get_attribute("image_size_x").as_int()
        self.image_h = self.camera_bp.get_attribute("image_size_y").as_int()
        self.camera_data = {'image': np.zeros((self.image_h, self.image_w, 4))}
        self.camera.listen(lambda data: self.create_image(data))

        self.collision = self.world.spawn_actor(self.collision_bp, self.lidar_init_transform, attach_to=self.tesla)
        self.collision_sensed = False
        self.collision.listen(lambda data: self.collided(data))

    def step(self, action):

        self.world.tick()

        done = False
        reward = 0

        self.step_counter += 1

        throttle = abs((action[0] - 10) / 10)
        steer = (action[1] - 10) / 10

        # print("Throttle:" + str(throttle))
        # print("Steer:" + str(steer))

        if action[0] >= 10:
            reverse = False
        else:
            reverse = True

        self.tesla.apply_control(carla.VehicleControl(steer=steer, reverse=reverse, throttle=throttle))

        # if not reverse:
        #
        #     velocity = self.tesla.get_velocity()
        #     # Velocity in m/s; 14 m/s is about 31.3 mph
        #     abs_velocity = math.sqrt(velocity.x**2 + velocity.y**2)
        #     if abs_velocity < 50:
        #         reward = abs_velocity/100
        #     else:
        #         reward = 0.5 - abs_velocity/100

        if not reverse:
            reward = (math.sqrt((self.tesla.get_location().x - self.init_location.x) ** 2 +
                                (self.tesla.get_location().y - self.init_location.y) ** 2) / 300)

            if Verbose and (self.step_counter % 500 == 0) and (self.step_counter != 0):
                print(reward)
                print("X Location:" + str(self.tesla.get_location().x))
                print("X Init Location:" + str(self.init_location.x))
                print("Y Location:" + str(self.tesla.get_location().y))
                print("Y Init Location:" + str(self.init_location.y))

        if reward >= 1:
            self.init_location = self.tesla.get_location()

        if self.collision_sensed:
            reward = -1
            done = True

        if self.tesla.get_location().z < -20:
            done = True

        if Verbose and (self.step_counter % 500 == 0) and (self.step_counter != 0):
            print("-------------------------------------")
            print("Throttle action: " + str(action[0]) + "\nThrottle: " + str(throttle) +
                  "\nSteering Action: " + str(action[1]) + "\nSteering: " + str(steer) + "\nReward: " + str(reward))

        if Show:
            cv.imshow('Top View', self.camera_data['image'])
            cv.imshow('Lidar View', np.dstack((self.blanks, self.blanks, self.lidar_data[1] * 255)))
            if Verbose and (self.step_counter % 100 == 0) and (self.step_counter != 0):
                print("Lidar Image Saved To File!")
                cv.imwrite(f"Output/{int(time.time())}.jpg", np.dstack((self.blanks, self.blanks, self.lidar_data[1] * 255)))
            cv.waitKey(1)

        return self.lidar_data[1], reward, False, done, {}

    def reset(self, **kwargs):

        self.collision_sensed = False

        self.tesla.apply_control(carla.VehicleControl(steer=0, reverse=False, throttle=0))
        self.tesla.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
        try:
            self.tesla.set_location(random.choice(self.spawn_points).location)
        finally:
            pass

        self.world.tick()

        self.init_location = self.tesla.get_location()

        if Verbose:
            print("Reset")

        return self.lidar_data[1], {}

    def create_lidar_plane(self, points):

        raw = np.copy(np.frombuffer(points.raw_data, dtype=np.dtype('f4')))

        x_points = (((Lidar_Field - 1) / 2) + np.around(raw[1::4] * Lidar_Resolution)).astype(dtype=int)
        y_points = (((Lidar_Field - 1) / 2) - np.around(raw[::4] * Lidar_Resolution)).astype(dtype=int)

        self.lidar_data[0, y_points, x_points] = 1
        self.lidar_index += len(x_points)

        if self.lidar_index >= 200:
            self.lidar_data[1] = self.lidar_data[0]
            self.lidar_data[0] = np.zeros((Lidar_Field, Lidar_Field), dtype=np.dtype('f4'))
            self.lidar_index = 0

        if ExtraVerbose and (self.step_counter % 100 == 0) and (self.step_counter != 0):
            print("x_points:" + str(len(x_points)))
            print("y_points:" + str(len(y_points)))
            print("index:" + str(self.lidar_index))

    def collided(self, data):
        self.collision_sensed = True

    def create_image(self, pic):
        if Show:
            self.camera_data['image'] = np.reshape(np.copy(pic.raw_data), (pic.height, pic.width, 4))

    def close(self):

        cv.destroyAllWindows()

        # destroys vehicles
        vehicles = self.world.get_actors().filter("*vehicle*")
        vehicles_len = len(vehicles)
        for i in range(vehicles_len):
            vehicles[i].destroy()
        # destroys sensors
        sensors = self.world.get_actors().filter("*sensor*")
        sensors_len = len(sensors)
        for i in range(sensors_len):
            sensors[i].destroy()
        # destroys walkers
        walkers = self.world.get_actors().filter("*walkers*")
        walkers_len = len(walkers)
        for i in range(walkers_len):
            walkers[i].destroy()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = None
        self.world.apply_settings(self.settings)
