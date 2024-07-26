# TODO: This does not work, even though step() is registering the action, the car does not move...
# This script is to allow you to play around with driving the car manually.
# It can be run from ControlPanel.py by setting Manual=True or run directly with all the defaults.
import cv2 as cv
from Env import Env
from Trainer import variableUnion


def ManualControl(*args):

    # There has got to be a better way to do this...
    setup = variableUnion(*list(args), library=[])
    setup.append({'Manual': True})
    env = Env(**setup[0])

    throttle = 0
    steer = 0
    brake = 0

    while True:

        keyed = cv.waitKey(1)

        if keyed == ord('w'):

            if throttle < 0.95:
                throttle = throttle + 0.1
                env.step([throttle, steer, brake])

        elif keyed == ord('s'):

            if throttle > -0.95:
                throttle = throttle - 0.1
                env.step([throttle, steer, brake])

        elif keyed == ord('d'):

            if steer < 0.975:
                steer = steer - 0.05
                env.step([throttle, steer, brake])

        elif keyed == ord('a'):

            if steer > -0.975:
                steer = steer - 0.05
                env.step([throttle, steer, brake])

        elif keyed == ord(' '):
            throttle = 0
            steer = 0
            brake = 1
            env.step([throttle, steer, brake])

        elif keyed == ord('q'):
            env.close()
            break
        else:
            env.step([throttle, steer, brake])


if __name__ == "__main__":
    ManualControl()

