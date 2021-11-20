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

    if len(sys.argv) < 2:
        print("Usage: python cassienet.py [option]", sys.argv)
        raise RuntimeError

    mode = sys.argv[1]
    sys.argv.remove(sys.argv[1])

    if mode == 'ppo':

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
        parser.add_argument("--do_prenorm",         default=False,         action='store_true') # Do pre-normalization or not

        
        parser.add_argument("--layers",             default="256,256",     type=str)            # hidden layer sizes in policy
        parser.add_argument("--arch",               default='ff')                               # either ff, lstm, or gru
        parser.add_argument("--bounded",            default=False,         type=bool)

        parser.add_argument("--workers",            default=2,             type=int)
        parser.add_argument("--redis",              default=None,          type=str)
        parser.add_argument("--previous",           default=None,          type=str)            # Dir of previously trained policy to start learning from

        args = parser.parse_args()

        # For curriculum learning
        #args = parse_previous(args)

        run_experiment(args)
    
    elif mode == 'rpo':
        """
            Utility for running Recursive Policy Optimization.

        """
        from algos.rpo import run_experiment

        parser.add_argument("--prenormalize_steps", default=100,                    type=int)
        parser.add_argument("--num_steps",          default=5000,                   type=int)
        parser.add_argument('--discount',           nargs='+', default=[0.99],      type=float) # the discount factor
        parser.add_argument("--learn_stddev",       default=False,         action='store_true') # learn std_dev or keep it fixed
        parser.add_argument('--std',                nargs='+', default=[0.13],      type=float) # the fixed exploration std
        parser.add_argument("--a_lr",               nargs='+', default=1e-4,        type=float) # adam learning rate for actor
        parser.add_argument("--c_lr",               nargs='+', default=1e-4,        type=float) # adam learning rate for critic
        parser.add_argument("--eps",                default=1e-6,                   type=float) # adam eps
        parser.add_argument("--kl",                 nargs='+', default=[0.02],      type=float) # kl abort threshold
        parser.add_argument("--entropy_coeff",      default=0.0,                    type=float)
        parser.add_argument("--grad_clip",          nargs='+', default=[0.05],      type=float)
        parser.add_argument("--batch_size",         nargs='+', default=[64],        type=int)   # batch size for policy update
        parser.add_argument("--epochs",             nargs='+', default=[3],         type=int)   # number of updates per iter
        parser.add_argument("--mirror",             nargs='+', default=[0],         type=float)
        parser.add_argument("--do_prenorm",         default=False,         action='store_true') # Do pre-normalization or not
        
        parser.add_argument("--planners",           nargs='+', default=None,        type=str)   # environment / robot to train on
        parser.add_argument("--paths",              nargs='+', default=None,        type=str)   # paths to policies. '' means no policy so learn one
        parser.add_argument("--planner_rate",       nargs='+', default=[2],         type=int)   # simrate of planners

        parser.add_argument("--layers",             nargs='+', default=["256,256"], type=str)   # hidden layer sizes in policy
        parser.add_argument("--arch",               nargs='+', default=['lstm'],    type=str)   # either ff, lstm, or gru
        parser.add_argument("--bounded",            nargs='+', default=[0],         type=int)   # whether or not output of policies is bounded
        parser.add_argument("--fixed_pol",          nargs='+', default=[0],         type=int)   # whether or not to fix training of policies at certain levels

        args = parser.parse_args()

        run_experiment(args)

    else:
        
        """Evaluation"""    
        parser.add_argument("--path",        type=str, default=None, help="path to folder containing policy and run details")
        parser.add_argument("--no_viz",      default=False, action='store_true')
        parser.add_argument("--interactive", default=False, action='store_true')
        parser.add_argument("--debug",       default=False, action='store_true')
        parser.add_argument("--pca",         default=False, action='store_true')
        parser.add_argument("--pds",         default=False, action='store_true')
        parser.add_argument("--gait",        default=False, action='store_true')
        parser.add_argument("--mirror",        default=False, action='store_true')

        args = parser.parse_args()

        if args.path is None:
            print("Must supply a --path argument.")
            exit(1)

        run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

        hrl = False
        files = [f for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))]
        for f in files:
            if 'actor_' in f:
                hrl = True

        if mode == 'udp':
            from envs.cassie.udp import run_udp
        
            policy = torch.load(args.path + "actor.pt")
            policy.eval()
            run_udp(policy, args, run_args)
            #eval_udp(args, run_args, hrl=hrl)
    
        elif mode == 'eval':
        
            if hrl: # hierarchical env
                if args.interactive:
                    from util.eval import hierarchical_interactive_eval
                    run_args.__dict__['path'] = args.path
                    hierarchical_interactive_eval(run_args, args)
                    exit()
                else:
                    from util.eval import hierarchical_simple_eval
                    run_args.__dict__['path'] = args.path
                    hierarchical_simple_eval(run_args, args.traj_len, args.mirror, terrain=args.terrain, stairs=args.stairs)
                    exit()
            else: # single-level env

                policy = torch.load(args.path + "actor.pt")
                policy.eval()
                #policy = pickle.load(open(args.path + "actor.pt", "rb"))
                if args.interactive:
                    from util.eval import EvalProcess_GestureCommands
                    ev = EvalProcess_GestureCommands(args, run_args)
                    ev.eval_policy(policy, terrain=args.terrain)
                else:
                    from util.eval import simple_eval
                    for key, val in vars(args).items():
                        if key not in run_args.__dict__:
                            run_args.__dict__[key] = val
                            print('Adding', key, 'to run_args.')
                    simple_eval(policy, run_args, args.traj_len, args.mirror, terrain=args.terrain, stairs=args.stairs)
