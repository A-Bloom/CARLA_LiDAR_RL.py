import random
import numpy as np
import cv2 as cv
import gymnasium as gym
from CleanUp import CleanUp
import carla
import math


# noinspection PyArgumentList
class BackEnv(gym.Env):

    Lidar_Depth = '128'
    Lidar_Resolution = 4
    Lidar_PPS = '9000'
    Lidar_RPS = '7'
    host = '127.0.0.1'
    port = 2000
    delta_seconds = 0.05
    Verbose = False
    Show = False
    Points_Per_Observation = 250

    def __init__(self):

        super(BackEnv, self).__init__()  # Initializes the base gymnasium class
        # For more information on the gymnasium class and how to inherit it see: https://gymnasium.farama.org/api/env/

        if self.Show:
            cv.namedWindow('Top View', cv.WINDOW_AUTOSIZE)  # Creates Top View camera window
            cv.namedWindow('Lidar View', cv.WINDOW_AUTOSIZE)  # Creates Lidar View lidar visualization window

        self.Lidar_Field = int(self.Lidar_Depth) * 2 * self.Lidar_Resolution + 1  # This is the full length that the
        # LiDAR covers. +1 for the car's location.

        # See Carla docs for carla functions. https://carla.readthedocs.io/en/0.9.15/
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        # TODO: Add a variable in the control panel to allow the user to specify the world or use a random world.
        self.world = self.client.get_world()  # Standard world
        # self.world = self.client.load_world(random.choice(self.client.get_available_maps()))  # Random world

        self.spectator = self.world.get_spectator()

        # These lines move the spectator automatically to get a better view.
        if self.spectator.get_transform().location.z < 150:
            self.spectator.set_transform(carla.Transform(carla.Location(x=50, y=50, z=250), carla.Rotation(pitch=-90)))

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.number_of_spawn_points = len(self.spawn_points)

        # Sets the world to synchronous mode so that the client controls when the world refreshes so that the agent
        # doesn't miss observations.
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.delta_seconds  # How many seconds the server simulates per frame.
        self.world.apply_settings(self.settings)

        # First create blueprints of everything, and then you can create instances of those blueprints.
        self.bps = self.world.get_blueprint_library()
        self.tesla_bp = self.bps.find('vehicle.tesla.model3')
        self.lidar_bp = self.bps.find('sensor.lidar.ray_cast')
        self.camera_bp = self.bps.find('sensor.camera.rgb')
        self.collision_bp = self.bps.find('sensor.other.collision')

        self.tesla_bp.set_attribute('color', '255, 110, 199')  # To make the agent highly visible.

        while True:  # Occasionally the agent can't spawn because there is something already there.
            try:
                self.tesla = self.world.spawn_actor(self.tesla_bp, random.choice(self.spawn_points))
                break
            except:
                print("Retrying Spawn")

        # Setting up some variables for action and reward function.
        self.abs_throttle = 0
        self.step_counter = 0
        self.init_location = self.tesla.get_location()
        self.target_location = random.choice(self.spawn_points).location
        self.distance_to_target = math.sqrt((self.target_location.x - self.init_location.x) ** 2 +
                                            (self.target_location.y - self.init_location.y) ** 2)

        # Sets up lidar attributes and attaches lidar to car.
        self.lidar_init_transform = carla.Transform(carla.Location(z=0.76))
        self.lidar_bp.set_attribute('channels', '1')
        self.lidar_bp.set_attribute('range', self.Lidar_Depth)
        self.lidar_bp.set_attribute('rotation_frequency', self.Lidar_RPS)
        self.lidar_bp.set_attribute('upper_fov', '0')
        self.lidar_bp.set_attribute('lower_fov', '0')
        self.lidar_bp.set_attribute('points_per_second', self.Lidar_PPS)
        self.lidar = self.world.spawn_actor(self.lidar_bp, self.lidar_init_transform, attach_to=self.tesla)
        # These are for LiDAR observations and visuals.
        self.lidar_data = np.zeros((2, self.Lidar_Field, self.Lidar_Field), dtype=np.uint8)
        self.blanks = np.zeros((self.Lidar_Field, self.Lidar_Field), dtype=np.uint8)
        self.blanks[round(self.Lidar_Field/2-0.5)][round(self.Lidar_Field/2-0.5)] = 1
        self.lidar_points = np.zeros((2, self.Points_Per_Observation, 2), dtype=np.float32)
        self.lidar_index = 0
        # LiDAR listener is in Env.py

        # Sets up camera attributes and attaches camera to car.
        self.camera_init_transform = carla.Transform(carla.Location(z=30), carla.Rotation(pitch=-90))
        self.camera_bp.set_attribute('image_size_x', '768')
        self.camera_bp.set_attribute('image_size_y', '768')
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_init_transform, attach_to=self.tesla)
        self.image_w = self.camera_bp.get_attribute("image_size_x").as_int()
        self.image_h = self.camera_bp.get_attribute("image_size_y").as_int()
        self.camera_data = np.zeros((self.image_h, self.image_w, 4))
        # Continuous data stream of images that triggers create_image().
        self.camera.listen(lambda data: self.create_image(data))

        # Sets up collision sensor attributes and attaches collision sensor to car.
        self.collision = self.world.spawn_actor(self.collision_bp, self.lidar_init_transform, attach_to=self.tesla)
        self.collision_sensed = False
        self.collision.listen(lambda data: self.collided(data))  # collided() triggered on collision.
        print("Initializing BackEnv")

    def step(self, action):
        return 0, 0, False, False, {}

    def reset(self, **kwargs):
        self.collision_sensed = False

        self.step_counter = 0

        # Respawns vehicle in new location.
        self.tesla.apply_control(carla.VehicleControl(steer=0, reverse=False, throttle=0))
        self.tesla.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
        try:
            self.tesla.set_location(random.choice(self.spawn_points).location)
        finally:
            pass

        # Must tick the world so the agent location gets updated before we retrieve the location for reward variables
        self.world.tick()

        # Resets reward variables.
        self.init_location = self.tesla.get_location()
        self.target_location = random.choice(self.spawn_points).location
        self.distance_to_target = math.sqrt((self.target_location.x - self.init_location.x) ** 2 +
                                            (self.target_location.y - self.init_location.y) ** 2)

        if self.Verbose:
            print("Reset")
            print(f"Distance to Target: {self.distance_to_target}")

        return 0, {}

    def close(self):
        cv.destroyAllWindows()
        CleanUp(self.world)

    def collided(self, data):
        self.collision_sensed = True

    def create_image(self, pic):
        if self.Show:
            self.camera_data = np.reshape(np.copy(pic.raw_data), (pic.height, pic.width, 4))

