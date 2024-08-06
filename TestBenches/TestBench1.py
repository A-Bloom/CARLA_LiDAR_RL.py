import argparse
import zipfile
import json

parser = argparse.ArgumentParser(
    description='Trains RL agents in CARLA. This can be called by a ControlPanel to start an experiment or through '
                'command line for the other options below.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-r', '--resume', help='Resume a truncated experiment run. Positional arguments will be '
                                          'automatically discarded.', dest='path_to_experiment')
group.add_argument('-u', '--upgrade', help='Allows you upgrade a single .zip policy. Must be followed by '
                                           'an epochs and a steps positional arguments.', dest='path_to_policy')
parser.add_argument('epochs', nargs='?', type=int)
parser.add_argument('steps', nargs='?', type=int)
args = parser.parse_args()

archive = zipfile.ZipFile(args.path_to_policy, 'r')
var_info = archive.open('var_info.json')
algorithm, experiment, algorithm_configuration = json.load(var_info)

print(algorithm)
