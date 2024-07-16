import Trainer

# Connection Options
# host = '10.230.117.122'
host = '127.0.0.1'
port = 2000
delta_seconds = 0.05  # Lower number leads to higher precision/longer train time

# Debugging Options
Show = True
Verbose = False

# LiDAR/Observation Options
Lidar_Depth = '30'
Lidar_Resolution = 1  # Points per meter
Lidar_PPS = '9000'  # Points/Second
Lidar_RPS = '7'  # Rotations/Second
observation_format = 'grid'  # 'points' or 'grid' or 'image'
Points_Per_Observation = 250

# Reward Options
reward_distribution = [0, 0.25, 0.5]  # [0 to 1 for speed, 0 to 1 for displacement, 0 to 1 for distance from target]
# (should add up to no more than 1 for normalization)
speed_limit = 50  # in m/s, 14 m/s is ~31.3 mph, the speed reward is normalized by this number
# so if you don't want a speed limit chose a number that you think will be the cars max speed.
min_speed = 1  # in m/s
min_speed_discount = -0.1  # will receive this if below min speed
displacement_reset = 200
turn_discount = 0.1  # 0 to 1

# Action Options
action_format = 'discrete'  # 'discrete' or 'continuous'
discrete_actions = 21  # Only for discrete steer and or throttle, must be odd!
action_possibilities = 1  # 0 for steer, 1 for throttle forward and steer,
# 2 for throttle and steer, 3 for throttle steer and break.
steer_cap = 1  # 0 to 1
throttle_cap = 1  # 0 to 1
constant_throttle = 0.5  # 0 to 1
turn_throttle_reduction = 0  # 0 to 1

