import carla

host = '127.0.0.1'
port = 2000

client = carla.Client(host, port)
client.set_timeout(20.0)

world = client.get_world()

# Releases world from synchronous mode.
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)

# Destroys sensors.
sensors = world.get_actors().filter("*sensor*")
for sensor in sensors:
    sensor.stop()
    sensor.destroy()
# Destroys vehicles.
vehicles = world.get_actors().filter("*vehicle*")
for vehicle in vehicles:
    vehicle.destroy()
# Destroys walkers.
walkers = world.get_actors().filter("*walkers*")
for walker in walkers:
    walker.destroy()

