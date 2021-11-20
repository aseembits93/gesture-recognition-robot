"""
Script for transferring policies trained under old version of cassienet to new versions
"""
import os
import argparse
import pickle

from collections import OrderedDict
import hashlib

def rename_experiment(logdir, new_name):
    os.rename(os.path.join(logdir, "experiment.pkl"), os.path.join(logdir, new_name+".pkl"))
    os.rename(os.path.join(logdir, "experiment.info"), os.path.join(logdir, new_name+".info"))
    print(f"Renamed old files to {new_name}.info and {new_name}.pkl")

def create_experiment(args, logdir):
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = args.__dict__
    assert "env" in arg_dict, \
    "You must provide a 'env' key in your command line arguments."

    # sort the keys so the same hyperparameters will always have the same hash
    arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

    # remove seed so it doesn't get hashed, store value for filename
    # same for logging directory
    run_name = arg_dict.pop('run_name')
    env_name = str(arg_dict['env'])

    os.makedirs(logdir, exist_ok=True)

    # Create a file with all the hyperparam settings in human-readable plaintext,
    # also pickle file for resuming training easily
    info_path = os.path.join(logdir, "experiment.info")
    pkl_path = os.path.join(logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(args, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')
    print("Wrote new files to experiment.info and experiment.pkl")


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
script_args = parser.parse_args()

run_args = pickle.load(open(os.path.join(script_args.path, "experiment.pkl"), "rb"))

if hasattr(run_args, "perception"):
    print("Looks like this policy has already been converted to new cassienet")
    ans = input("Sure you want to continue with conversion? (y/n)")
    if ans.lower() == "n":
        exit()

# rename old files
rename_experiment(script_args.path, "experiment_bkup")

args = argparse.Namespace()

YESH = False

"""Environment"""
args.simrate = run_args.simrate
if 'dyn_random' in run_args:
    args.dynamics_randomization = run_args.dyn_random
else:
    args.dynamics_randomization = False
args.impedance = run_args.impedance
if YESH:
    args.incentive = False if "no_incentive" in run_args.reward else True  # For Yesh
    args.phase_std = run_args.behavior_std  # For Yesh
else:
    args.incentive = False
    args.phase_std = 0.25
# These are hardcoded based on policy being converted.
args.standing = False
args.fixed_hop = False
args.fixed_walk = False
args.reward = 'consolidated'
args.task = 'speed'
args.perception = False
args.stairs = False
args.height = False
args.gaze_control = False

"""Logger / Saver"""
args.logdir = run_args.logdir
args.env = "cassie"
args.run_name = run_args.run_name

"""Curriculum Learning"""
args.exchange_reward = run_args.exchange_reward
args.previous = run_args.previous

"""All RL algorithms"""
args.nolog = run_args.nolog
args.arch = run_args.arch
args.seed = run_args.seed
args.traj_len = run_args.traj_len
args.layers = run_args.layers
args.timesteps = run_args.timesteps

create_experiment(args, script_args.path)
