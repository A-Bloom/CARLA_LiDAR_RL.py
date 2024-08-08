import torch
from stable_baselines3.common import utils

if torch.cuda.device_count() > 1:
    device = 'cuda:1'
else:
    device = 'auto'

print(utils.get_device(device))
