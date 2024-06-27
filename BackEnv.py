import random
import __main__
from pathlib import Path
import numpy as np
import cv2 as cv
import gymnasium as gym
import carla


# noinspection PyArgumentList
class BackEnv(gym.Env):

    Lidar_Depth = '128'
    Lidar_Resolution = 4
    Lidar_PPS = '9000'
    Lidar_RPS = '7'
    host = '127.0.0.1'
    port = 2000
    delta_seconds = 0.05
    Verbose = True
    Show = True

    def __init__(self):

        super(BackEnv, self).__init__()

        cv.namedWindow('Top View', cv.WINDOW_AUTOSIZE)
        cv.namedWindow('Lidar View', cv.WINDOW_AUTOSIZE)

        self.Lidar_Field = int(self.Lidar_Depth) * 2 * self.Lidar_Resolution + 1
        self.Points_Per_Observation = int(int(self.Lidar_PPS) / int(self.Lidar_RPS))

        self.client = carla.Client(self.host, self.port)
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
        self.settings.fixed_delta_seconds = self.delta_seconds
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
        self.lidar_bp.set_attribute('range', self.Lidar_Depth)
        self.lidar_bp.set_attribute('rotation_frequency', self.Lidar_RPS)
        self.lidar_bp.set_attribute('upper_fov', '0')
        self.lidar_bp.set_attribute('lower_fov', '0')
        self.lidar_bp.set_attribute('points_per_second', self.Lidar_PPS)
        self.lidar = self.world.spawn_actor(self.lidar_bp, self.lidar_init_transform, attach_to=self.tesla)
        self.lidar_data = np.zeros((2, self.Lidar_Field, self.Lidar_Field), dtype=np.float32)
        self.blanks = np.zeros((self.Lidar_Field, self.Lidar_Field))
        self.lidar_index = 0
        self.lidar.listen(lambda data: self.create_lidar_plane(data))

        self.camera_init_transform = carla.Transform(carla.Location(z=30), carla.Rotation(pitch=-90))
        self.camera_bp.set_attribute('image_size_x', '768')
        self.camera_bp.set_attribute('image_size_y', '768')
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_init_transform, attach_to=self.tesla)
        self.image_w = self.camera_bp.get_attribute("image_size_x").as_int()
        self.image_h = self.camera_bp.get_attribute("image_size_y").as_int()
        self.camera_data = np.zeros((self.image_h, self.image_w, 4))
        self.camera.listen(lambda data: self.create_image(data))

        self.collision = self.world.spawn_actor(self.collision_bp, self.lidar_init_transform, attach_to=self.tesla)
        self.collision_sensed = False
        self.collision.listen(lambda data: self.collided(data))
        print("BackEnv initializing")

    def step(self, action):
        return 0, 0, False, False, {}

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

        if self.Verbose:
            print("Reset")

        return 0, {}

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

    def collided(self, data):
        self.collision_sensed = True

    def create_image(self, pic):
        if self.Show:self.camera_data = np.reshape(np.copy(pic.raw_data), (pic.height, pic.width, 4))

    def name(self):
        return Path(__main__.__file__).stem
