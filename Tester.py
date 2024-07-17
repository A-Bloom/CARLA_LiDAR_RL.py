import MidEnv
import carla

Env = MidEnv.MidEnv()

while True:
    Env.step([0, 0, 0])
