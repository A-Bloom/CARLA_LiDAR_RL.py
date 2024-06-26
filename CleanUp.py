import carla

host = '127.0.0.1'

port = 2000

client = carla.Client(host, port)
client.set_timeout(20.0)

world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)

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

