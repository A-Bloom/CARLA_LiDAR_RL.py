from pathlib import Path
import sys
import __main__

the_path = r"C:\Users\abche\Documents\F1_10_Mini_Autonomous_Driving\LIDAR_1\Output\Experiment_07_19_09_09\A2C\07_19_09_09_41"

print(the_path)

file = Path(the_path)

print(file.exists())
print(file.is_file())
print(file.is_dir())

for arg in sys.argv[1:]:
    print(arg)
