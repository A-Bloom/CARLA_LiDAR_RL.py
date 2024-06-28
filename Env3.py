from pathlib import Path
from MidEnv import MidEnv


class CarEnv(MidEnv):

    def __init__(self):

        # Connection Options
        # host = '10.230.117.122'
        self.host = '127.0.0.1'
        self.port = 2000
        self.delta_seconds = 0.05  # Lower number leads to higher precision/longer train time

        # Debugging Options
        self.Show = True
        self.Verbose = True

        # LiDAR/Observation Options
        self.Lidar_Depth = '30'
        self.Lidar_Resolution = 4  # Points per meter
        self.Lidar_PPS = '9000'  # Points/Second
        self.Lidar_RPS = '7'  # Rotations/Second
        self.observation_format = 'grid'  # 'points' or 'grid'
        self.Points_Per_Observation = 250

        # Reward Options
        self.reward_distribution = [0.75, 0.25]  # [0 to 1 for speed, 0 to 1 for displacement]
        # (should add up to no more than 1 for normalization)
        self.speed_limit = 40  # in m/s, 14 m/s is ~31.3 mph, the speed reward is normalized by this number
        # so if you don't want a speed limit chose a number that you think will be the cars max speed.
        self.displacement_reset = 200
        self.turn_discount = 0.1  # 0 to 1

        # Action Options
        self.action_format = 'continuous'  # 'discrete' or 'continuous'
        self.discrete_actions = 21  # Only for discrete steer and or throttle, must be odd!
        self.action_possibilities = 3  # 0 for steer, 1 for throttle forward and steer,
        # 2 for throttle and steer, 3 for throttle steer and break.
        self.steer_cap = 1  # 0 to 1
        self.throttle_cap = 1  # 0 to 1
        self.constant_throttle = 0.5  # 0 to 1
        self.turn_throttle_reduction = 0  # 0 to 1

        super(CarEnv, self).__init__()

    def step(self, action):
        super(CarEnv, self).step(action)
        return self.observation, self.reward, False, self.done, {}

    def reset(self, **kwargs):
        super(CarEnv, self).reset()
        return self.observation, {}

    def close(self):
        super(CarEnv, self).close()

    def name(self):
        return Path(__file__).stem
