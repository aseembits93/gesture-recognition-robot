import sys
import os
import pickle
import argparse

import torch

from util.env import env_factory
from util.log import create_logger
from util.logo import color, print_logo

import nn

def print_logo(subtitle="", option=2):
    print()
    print(color.BOLD + color.ORANGE +  "                                                               ╱▔▔▔▔▔▔╲   ")
    print(color.BOLD + color.ORANGE +  "                                                             ▂╱ ╱▔╲    ╲  ")
    print(color.BOLD + color.ORANGE +  "                                                            ╱╲ ▕   ▏╲ ╲▕  ")
    print(color.BOLD + color.ORANGE +  "                                                           ╱ ▕▂▂▏  ╲╱▏▕▔  ")
    print(color.BOLD + color.ORANGE +  "                     _                                     ▏ ╱▂▂╲    ╲╱   ")
    print(color.BOLD + color.ORANGE +  "                    | |                                 ▕╲ ▋╱ ╱ ▕         ")
    print(color.BOLD + color.ORANGE +  " _ __ ___   __ _  __| |_ __ _   _ _ __  _ __   ___ _ __  ▏▔▔ ╱  ╱╲        ")
    print(color.BOLD + color.ORANGE +  "| '__/ _ \ / _` |/ _` | '__| | | | '_ \| '_ \ / _ \ '__| ╲▂▂╱▔▔▔╲ ╲       ")
    print(color.BOLD + color.ORANGE +  "| | | (_) | (_| | (_| | |  | |_| | | | | | | |  __/ |            ╲ ╲      ")
    print(color.BOLD + color.ORANGE +  "|_|  \___/ \__,_|\__,_|_|   \__,_|_| |_|_| |_|\___|_|             ╲ ╲     "+ color.END)
    print("\n")
    print(subtitle)
    print("\n")

if __name__ == "__main__":


    print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")
    parser = argparse.ArgumentParser()


    """Environment"""
    parser.add_argument("--not_dyn_random", dest='dynamics_randomization', default=True, action='store_false')
    parser.add_argument("--impedance",  default=False, action='store_true', )
    parser.add_argument("--incentive",  default=False, action='store_true', )
    parser.add_argument("--standing",   default=False, action='store_true', )
    parser.add_argument("--fixed_hop",  default=False, action='store_true', )
    parser.add_argument("--fixed_walk", default=False, action='store_true', )
    parser.add_argument("--phase_std",  default=0.1, type=float)
    parser.add_argument("--task",       default='speed')
    parser.add_argument("--perception", default=False, action='store_true', )
    parser.add_argument("--stairs",     default=False, action='store_true', )
    parser.add_argument("--stair_quat", default=False,    action='store_true')          # randomize tilt of stairs
    parser.add_argument("--stair_slats", default=False,    action='store_true')         # randomize whether stairs are slatted or not
    parser.add_argument("--height", default=False, action='store_true', )
    parser.add_argument("--gaze_control", default=False, action='store_true', )
    parser.add_argument("--terrain",  default=False, action='store_true')
    parser.add_argument("--env",      default="cassie", type=str)                     # environment to train on
    parser.add_argument("--reward", default="icra", type=str)                                         # reward to use. this is a required argument.
    parser.add_argument("--simrate",   default=50, type=int, help="simrate of environment")


    """Logger / Saver"""
    parser.add_argument("--wandb",    default=False, action='store_true')              # use weights and biases for training
    parser.add_argument("--wandb_project_name",    default="roadrunner")              # use weights and biases for training
    parser.add_argument("--logdir",   default="./trained_models/", type=str)
    parser.add_argument("--run_name", default=None)                                                 # run name


    """All RL algorithms"""
    parser.add_argument("--nolog",     action='store_true')                            # store log data or not.
    parser.add_argument("--seed",      default=0,           type=int)                  # random seed for reproducibility
    parser.add_argument("--traj_len",  default=1000,        type=int)                  # max trajectory length for environment
    parser.add_argument("--timesteps", default=1e8,         type=float)                # timesteps to run experiment for


    """
        Utility for running Proximal Policy Optimization.

    """
    from algos.ppo import run_experiment
    parser.add_argument("--prenormalize_steps", default=100,           type=int)
    parser.add_argument("--num_steps",          default=5000,          type=int)
    parser.add_argument('--discount',           default=0.99,          type=float)          # the discount factor
    parser.add_argument("--learn_stddev",       default=False,         action='store_true') # learn std_dev or keep it fixed
    parser.add_argument('--std',                default=0.13,          type=float)          # the fixed exploration std
    parser.add_argument("--a_lr",               default=1e-4,          type=float)          # adam learning rate for actor
    parser.add_argument("--c_lr",               default=1e-4,          type=float)          # adam learning rate for critic
    parser.add_argument("--eps",                default=1e-6,          type=float)          # adam eps
    parser.add_argument("--kl",                 default=0.02,          type=float)          # kl abort threshold
    parser.add_argument("--entropy_coeff",      default=0.0,           type=float)
    parser.add_argument("--clip",               default=0.2,           type=float)          # Clipping parameter for PPO surrogate loss
    parser.add_argument("--grad_clip",          default=0.05,          type=float)
    parser.add_argument("--batch_size",         default=64,            type=int)            # batch size for policy update
    parser.add_argument("--epochs",             default=3,             type=int)            # number of updates per iter
    parser.add_argument("--mirror",             default=0,             type=float)
    
    parser.add_argument("--layers",             default="256,256",     type=str)            # hidden layer sizes in policy
    parser.add_argument("--arch",               default='ff')                               # either ff, lstm, or gru
    parser.add_argument("--bounded",            default=False,         type=bool)

    parser.add_argument("--workers",            default=2,             type=int)
    parser.add_argument("--redis",              default=None,          type=str)

    args = parser.parse_args()
    # For curriculum learning
    #args = parse_previous(args)

    args.env = "cassie_FFWalk"
    args.wandb = True
    args.run_name = "3Step_test_run-02-2_seed{}".format(args.seed)
    args.logdir = "./logs/dump"

    args.run_name = "test"
    args.logdir = "./logs/dump"

    args.layers = "128,128"
    args.discount = 0.95
    args.mirror = 1
    args.batch_size = 64
    args.num_steps = 50000
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.epochs = 5
    args.traj_len = 300
    args.timesteps = 4e9
    args.workers = 56
    args.arch = "ff"
    args.fixed_walk = True

    args.reward = "icra"
    args.do_prenorm = False
    # args.prenormalize_steps = 10000
    # args.previous = "./pretrained_models/speed_base_fixedwalk_dx05to15_sf09to13/"


    run_experiment(args)
    
    