import os
import sys

algorithm = 'PPO'

folder_name = sys.argv[1]

model_dir = max(os.listdir(f"{folder_name}/{algorithm}"))
model_names = os.listdir(f"{folder_name}/{algorithm}/{model_dir}")
model_names.sort()
model_path = f"{folder_name}/{algorithm}/{model_dir}/{model_names[-2]}"

print(model_path)
