import glob
import os
import sys
import time
import random
import math
import numpy as np
import cv2 as cv
import subprocess


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

host = '127.0.0.1'  # localhost
# host = '128.252.83.117'  # External IP
# host = '10.230.117.122'  # Internal IP
port = 2000

client = carla.Client(host, port)
client.set_timeout(20.0)


world = client.load_world(random.choice(client.get_available_maps()))
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)

bps = world.get_blueprint_library()
tesla_bp = bps.find('vehicle.tesla.model3')
lidar_bp = bps.find('sensor.lidar.ray_cast')
camera_bp = bps.find('sensor.camera.rgb')
collision_bp = bps.find('sensor.other.collision')

tesla_bp.set_attribute('color', '(252, 15, 192)')
tesla = world.spawn_actor(tesla_bp, random.choice(spawn_points))

lidar_init_transform = carla.Transform(carla.Location(z=2))
camera_init_transform = carla.Transform(carla.Location(z=30), carla.Rotation(pitch=-90))

lidar_bp.set_attribute('channels', '1')
lidar_bp.set_attribute('range', '128')
lidar_bp.set_attribute('rotation_frequency', '7')
lidar_bp.set_attribute('upper_fov', '0')
lidar_bp.set_attribute('lower_fov', '0')
lidar_bp.set_attribute('points_per_second', '9000')

lidar = world.spawn_actor(lidar_bp, lidar_init_transform, attach_to=tesla)

camera_bp.set_attribute('image_size_x', '768')
camera_bp.set_attribute('image_size_y', '768')

camera = world.spawn_actor(camera_bp, camera_init_transform, attach_to=tesla)

collision = world.spawn_actor(collision_bp, lidar_init_transform, attach_to=tesla)

init_location = tesla.get_location()


def create_image(pic, data_dict):
    data_dict['image'] = np.reshape(np.copy(pic.raw_data), (pic.height, pic.width, 4))


image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
camera_data = {'image': np.zeros((image_h, image_w, 4))}

camera.listen(lambda data: create_image(data, camera_data))

cv.namedWindow('Top View', cv.WINDOW_AUTOSIZE)
cv.imshow('Top View', camera_data['image'])
cv.waitKey(1)


def create_lidar_plane(points, data_dict):
    data = np.copy(np.frombuffer(points.raw_data, dtype=np.dtype('f4')))
    i = 0
    data_len = len(data)
    while i < data_len:
        if data_dict['index'] == 500:
            data_dict['points'] = np.ones((256, 256, 3))
            data_dict['points'][128, 128] = [0, 0, 0]
            data_dict['index'] = 0

        data_dict['points'][(128-int(data[i])), (int(data[i+1])+128)] = [0, 0, 255]
        data_dict['index'] = data_dict['index']+1
        i = i+4


lidar_data = {'points': np.ones((256, 256, 3)), 'index': 0}

lidar.listen(lambda data: create_lidar_plane(data, lidar_data))


def collided(collision_data):
    print("Collision Detected!")


collision.listen(lambda data: collided(data))

abs_throttle = 0

cv.namedWindow('Lidar View', cv.WINDOW_AUTOSIZE)
cv.imshow('Lidar View', lidar_data['points'])
cv.waitKey(1)

while True:
    cv.imshow('Top View', camera_data['image'])
    cv.imshow('Lidar View', lidar_data['points'])

    try:
        world.tick()
    except RuntimeError:
        print("Connection to Server Lost Attempting Reboot...")
        subprocess.Popen(r"C:\Users\abche\Documents\F1_10_Mini_Autonomous_Driving\CARLA_0.9.15\CarlaUE4.exe")
        time.sleep(5)
        try:
            world.tick()
        except RuntimeError:
            print("Failed to Reconnect to Server, Shutting Down.")
            keyed = ord('q')

    keyed = cv.waitKey(1)
    if keyed == ord('w'):

        if abs_throttle < 0.95:
            throttle = tesla.get_control().throttle
            abs_throttle = abs_throttle+0.1
            steer = tesla.get_control().steer

            if abs_throttle >= 0:
                throttle = throttle + 0.1
                reverse = False
            else:
                throttle = throttle - 0.1
                reverse = True

            tesla.apply_control(carla.VehicleControl(reverse=reverse, throttle=throttle, steer=steer))

    elif keyed == ord('s'):

        if abs_throttle > -0.95:
            throttle = tesla.get_control().throttle
            abs_throttle = abs_throttle-0.1
            steer = tesla.get_control().steer

            if abs_throttle <= 0:
                throttle = throttle + 0.1
                reverse = True
            else:
                throttle = throttle - 0.1
                reverse = False

            tesla.apply_control(carla.VehicleControl(reverse=reverse, throttle=throttle, steer=steer))

    elif keyed == ord('d'):

        steer = tesla.get_control().steer
        if steer < 0.975:
            steer = steer + 0.05
            reverse = tesla.get_control().reverse
            throttle = tesla.get_control().throttle

            tesla.apply_control(carla.VehicleControl(steer=steer, reverse=reverse, throttle=throttle))

    elif keyed == ord('a'):

        steer = tesla.get_control().steer
        if steer > -0.975:
            steer = steer - 0.05
            reverse = tesla.get_control().reverse
            throttle = tesla.get_control().throttle

            tesla.apply_control(carla.VehicleControl(steer=steer, reverse=reverse, throttle=throttle))

    elif keyed == ord(' '):
        abs_throttle = 0
        tesla.apply_control(carla.VehicleControl(throttle=0, brake=1.0, steer=0))

    elif keyed == ord('p'):
        velocity = tesla.get_velocity()
        # Velocity in m/s; 14 m/s is about 31.3 mph
        abs_velocity = math.sqrt(velocity.x**2 + velocity.y**2)
        print("The cars Velocity is:")
        print(abs_velocity)
        print("throttle:")
        print(tesla.get_control().throttle)
        print("abs_throttle:")
        print(abs_throttle)
        distance = (math.sqrt((tesla.get_location().x - init_location.x) ** 2 +
                            (tesla.get_location().y - init_location.y) ** 2))
        print("Distance Traveled: " + str(distance) + " meters")
        print("X Location:" + str(tesla.get_location().x))
        print("X Init Location:" + str(init_location.x))
        print("Y Location:" + str(tesla.get_location().y))
        print("Y Init Location:" + str(init_location.y))


    elif keyed == ord('r'):
        tesla.set_location(random.choice(spawn_points).location)
        tesla.apply_control(carla.VehicleControl(throttle=0, steer=0))
        abs_throttle = 0
        time.sleep(1)
        init_location = tesla.get_location()

    elif keyed == ord('m'):

        vehicles = world.get_actors().filter("*vehicle*")
        vehicles_len = len(vehicles)
        for i in range(vehicles_len):
            vehicles[i].destroy()

        sensors = world.get_actors().filter("*sensor*")
        sensors_len = len(sensors)
        for i in range(sensors_len):
            sensors[i].destroy()

        walkers = world.get_actors().filter("*walkers*")
        walkers_len = len(walkers)
        for i in range(walkers_len):
            walkers[i].destroy()

        world = client.load_world(random.choice(client.get_available_maps()))
        spawn_points = world.get_map().get_spawn_points()

        tesla = world.spawn_actor(tesla_bp, random.choice(spawn_points))
        lidar = world.spawn_actor(lidar_bp, lidar_init_transform, attach_to=tesla)
        camera = world.spawn_actor(camera_bp, camera_init_transform, attach_to=tesla)
        collision = world.spawn_actor(collision_bp, lidar_init_transform, attach_to=tesla)
        collision.listen(lambda data: collided(data))
        lidar.listen(lambda data: create_lidar_plane(data, lidar_data))
        camera.listen(lambda data: create_image(data, camera_data))

    elif keyed == ord('q'):
        break


cv.destroyAllWindows()

# destroys vehicles
vehicles = world.get_actors().filter("*vehicle*")
vehicles_len = len(vehicles)
for i in range(vehicles_len):
    vehicles[i].destroy()
# destroys sensors
sensors = world.get_actors().filter("*sensor*")
sensors_len = len(sensors)
for i in range(sensors_len):
    sensors[i].destroy()
# destroys walkers
walkers = world.get_actors().filter("*walkers*")
walkers_len = len(walkers)
for i in range(walkers_len):
    walkers[i].destroy()


# Reset world to asynchronous mode
settings.synchronous_mode = False
world.apply_settings(settings)

