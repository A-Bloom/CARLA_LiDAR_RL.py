# This script is to allow you to play around with driving the car manually.
# It can be run from ControlPanel.py by setting Manual=True or run directly with all the defaults.
# a, s, d, w for throttle and steer. q to exit.
import cv2 as cv
from Env import Env
from Trainer import variableUnion


def ManualControl(*args):

    # There has got to be a better way to do this...
    setup = variableUnion(*list(args), library=[])
    setup.append({'Manual': True})
    setup[0]['action_format'] = 'continuous'
    setup[0]['action_possibilities'] = 3
    env = Env(**setup[0])

    manual_throttle = 0
    manual_steer = 0
    manual_brake = 0

    while True:

        keyed = cv.waitKey(1)

        manual_brake = 0

        if keyed == ord('w'):

            if manual_throttle < 0.95:
                manual_throttle = manual_throttle + 0.1
                env.step([manual_throttle, manual_steer, manual_brake])

        elif keyed == ord('s'):

            if manual_throttle > -0.95:
                manual_throttle = manual_throttle - 0.1
                env.step([manual_steer, manual_throttle, manual_brake])

        elif keyed == ord('d'):

            if manual_steer < 0.975:
                manual_steer = manual_steer + 0.05
                env.step([manual_steer, manual_throttle, manual_brake])

        elif keyed == ord('a'):

            if manual_steer > -0.975:
                manual_steer = manual_steer - 0.05
                env.step([manual_steer, manual_throttle, manual_brake])

        elif keyed == ord(' '):
            manual_throttle = 0
            manual_steer = 0
            manual_brake = 1
            env.step([manual_steer, manual_throttle, manual_brake])

        elif keyed == ord('q'):
            env.close()
            break
        else:
            env.step([manual_steer, manual_throttle, manual_brake])


if __name__ == "__main__":
    ManualControl()

