from typing import ForwardRef
import torch
import hashlib, os, pickle
from collections import OrderedDict
import sys, time

from datetime import datetime

import multiprocessing as mp  # for live plots

import os
import tty
import termios
import select
import numpy as np
from util.env import env_factory
from util.mirror import mirror_tensor
from copy import deepcopy

#from util.parse_hierarchical_args import parse_envs
from nn.hierarchy import RecursiveSystem

def parse_envs(args):
    envs = args.planners
    base_env = args.env
    env_fn_stack = []
    for env in envs:
        args.__dict__['env'] = env
        env_fn_stack += [env_factory(**vars(args))]
    args.__dict__['env'] = 'cassie'
    env_fn_stack += [env_factory(**vars(args))]

    return env_fn_stack

def create_recursive_system(run_args, mirror=False, terrain=False, stairs=False, interactive_mode=False):
    run_args.dynamics_randomization = False
    run_args.evaluation_mode = True
    if interactive_mode:
        run_args.interactive_mode = True
    run_args.__dict__['terrain'] = terrain
    run_args.__dict__['stairs'] = stairs
    env_fn_stack = parse_envs(run_args)

    d = len(env_fn_stack)-1
    base_envname = run_args.env

    fixed_pol = [True for _ in range(d)]
    # fixed_pol = [False, True]
    
    actors = []
    critics = []
    for p in run_args.planners:
        suffix = p + '-' + base_envname
        actors  += [torch.load(os.path.join(run_args.path, f'actor_{suffix}.pt'))]
        critics += [torch.load(os.path.join(run_args.path, f'critic_{suffix}.pt'))]
    return RecursiveSystem(actors, critics, env_fn_stack, fixed_pol)

def hierarchical_simple_eval(run_args, traj_len, mirror=False, terrain=False, stairs=False):
    system = create_recursive_system(run_args, mirror=mirror, terrain=terrain, stairs=stairs)
    while True:
        print(system.eval(trials=1, max_len=traj_len, render=True, mirror=mirror))


def hierarchical_interactive_eval(run_args, args, terrain):
    system = create_recursive_system(run_args, mirror=args.mirror, terrain=terrain, stairs=args.stairs, interactive_mode=True)

    # if len([x for x in [args.gait, args.pca, args.pds] if x is True]) > 1:
    #     print("right now multiple live plots does not seem to work.")
    #     exit(1)

    # if args.gait:
    #     from .gait_plot import Live_Gait_Plot
    #     gait_plot_pipe, gait_plotter_pipe = mp.Pipe()
    #     gait_plotter = Live_Gait_Plot(locomotion_env.behavior)
    #     gait_plot_process = mp.Process(target=gait_plotter, args=(gait_plotter_pipe,))
    #     gait_plot_process.start()
    #     send_gait = gait_plot_pipe.send
    #     print("STARTING Gait PLOT")
    # if args.pca:
    #     from util.pca import Live_PCA_Plot, get_hiddens
    #     pca_plot_pipe, pca_plotter_pipe = mp.Pipe()
    #     pca_plotter = Live_PCA_Plot(locomotion_policy, locomotion_env)
    #     pca_plot_process = mp.Process(target=pca_plotter, args=(pca_plotter_pipe,))
    #     pca_plot_process.start()
    #     send_pca = pca_plot_pipe.send
    #     get_hiddens = get_hiddens
    #     print("STARTING PCA PLOT")
    # if args.pds:
    #     from util.pds import Live_PD_Plot
    #     pd_plot_pipe, pd_plotter_pipe = mp.Pipe()
    #     pd_plotter = Live_PD_Plot()
    #     pd_plot_process = mp.Process(target=pd_plotter, args=(pd_plotter_pipe,))
    #     pd_plot_process.start()
    #     send_pd = pd_plot_pipe.send
    #     print("STARTING PD PLOT")

    old_settings = termios.tcgetattr(sys.stdin)

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    with torch.no_grad():
        try:
            tty.setcbreak(sys.stdin.fileno())
            evals = np.zeros(len(system.policy_stack))
            steps = 0
            done = False
            mirror = args.mirror
            system.reset()
            
            while True:

                keyin = None

                if isData():

                    keyin = sys.stdin.read(1)
                    if keyin == 'r':
                        system.reset()
                        print("Resetting Control Hierarchy!")
                    if keyin == 'm':
                        mirror = not mirror
                        print(f"Mirror: {mirror}")
                
                recursive_sample = system.step(deterministic=True, render=True, update_norm=False, mirror=mirror, keypress=keyin)
                os.system('clear')
                print(system.env_head.log_cmd())
                print()
                lens = []
                for depth in range(len(system.policy_stack)):
                    #print("recursive sample at depth {} is: {}".format(depth+1, recursive_sample))
                    states, actions, rewards, dones, next_level = recursive_sample
                    evals[depth] += np.sum(rewards)
                    level_done = done or dones if isinstance(dones, bool) else True in dones
                    if level_done:
                        done = True
                    recursive_sample = next_level
                    lens.append(1 if depth == 0 else len(states))
                steps += max(lens)
            
            # print(f"Eval reward: {evals / trials}")
        
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            time.sleep(0.02)
            # if args.pca:
            #     pca_plot_process.join()
            # if args.pds:
            #     pd_plot_process.join()
            # if args.gait:
            #     gait_plot_process.join()
        

def simple_eval(policy, run_args, max_len, mirror=False, stairs=False, terrain=None, **kwargs):
    run_args.__dict__['stairs'] = run_args.stairs or stairs
    env = env_factory(**vars(run_args))()
    import pickle
    from functools import partial
    env.dynamics_randomization = False
    with torch.no_grad():
        ep_returns = []
        if mirror:
            m_idx = env.get_state_mirror_indices()
            a_idx = env.get_action_mirror_indices()
        for traj in range(100):
            state = env.reset()
            done = False
            traj_len = 0
            ep_return = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            while not done and traj_len < max_len:
                env.speed = 1
                env.side_speed = 0
                env.orient_add = 0
                env.cmd_dict["x_vel"] = 2.0
                env.cmd_dict["y_vel"] = 0.0
                state = torch.tensor(state).float()
                if mirror:
                    state = mirror_tensor(state, m_idx)

                action = policy(state)
                if mirror:
                    action = mirror_tensor(action, a_idx)
                action = action.numpy()
                next_state, reward, done, _ = env.step(action, render=True)
                env.render()
                state = next_state
                ep_return += reward
                traj_len += 1
            ep_returns += [ep_return]
            print('Return:', ep_return)
    return np.mean(ep_returns)

def simple_eval_with_logging(policy, run_args, max_len, mirror=False, stairs=False, terrain=None, **kwargs):
    run_args.__dict__['stairs'] = run_args.stairs or stairs
    env = env_factory(**vars(run_args))()
    env.dynamics_randomization = False
    cassie_pose_array = np.empty([len(np.arange(0.1,2.1,0.1)), 1], dtype=object)
    target_speed_enum = 0
    #print(cassie_pose_array)
    target_velocity = 0.1
    import pickle
    from functools import partial
    with torch.no_grad():
        ep_returns = []
        if mirror:
            m_idx = env.get_state_mirror_indices()
            a_idx = env.get_action_mirror_indices()
        while target_velocity < 2.1:
            state = env.reset()

            done = False
            traj_len = 0
            ep_return = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            
            listOfPoses = []
            
            while not done and traj_len < max_len:
                env.speed = 1
                env.side_speed = 0
                env.orient_add = 0
                env.cmd_dict["x_vel"] = target_velocity
                env.cmd_dict["y_vel"] = 0.0
                state = torch.tensor(state).float()
                if mirror:
                    state = mirror_tensor(state, m_idx)

                action = policy(state)

                if mirror:
                    action = mirror_tensor(action, a_idx)

                action = action.numpy()


                next_state, reward, done, logData, _ = env.step(action, render=False, logging = True)
                poseDict = {}

                for count, row in enumerate(logData, start = 0):
                    poseDict["qpos"] = row[1:36]
                    poseDict["qvel"] = row[37:69]
                    poseDict["leftFootGRF_Z"] = row[-7]
                    poseDict["rightFootGRF_Z"] = row[-5]
                    poseDict["pelvisVel"] = row[-3]
                    poseDict["phaseAngle"] = row[-1]

                    #cassie_pose_array[target_speed_enum, (traj_len*50)+count, :] = poseDict
                    listOfPoses.append(poseDict)
                
                #env.render()
                state = next_state
                ep_return += reward
                traj_len += 1
            
            listOfPoses = listOfPoses[-(220*50):]
            cassie_pose_array[target_speed_enum, 0] = listOfPoses
            ep_returns += [ep_return]
            print('Return:', ep_return)
            target_velocity += 0.1
            target_speed_enum += 1

        #startIndex = 0
        #endIndex = 0

        for i in range(cassie_pose_array.shape[0]):
            startIndex = 0
            endIndex = 0
            for count, element in enumerate(cassie_pose_array[i,0]):
                phaseAngle = element["phaseAngle"]
                leftFootGRF = element["leftFootGRF_Z"]
                rightFootGRF = element["rightFootGRF_Z"]

                if((phaseAngle < 50) and (leftFootGRF == 0)):
                    startIndex = count
                    break 

            #print(len(cassie_pose_array[i,0]))
            #print(len(cassie_pose_array[i,0]))
            #print(cassie_pose_array[i,0][0])
            cassie_pose_array[i,0] = cassie_pose_array[i,0][startIndex:]
            for count, element in enumerate(cassie_pose_array[i,0]):
                phaseAngle = element["phaseAngle"]
                leftFootGRF = element["leftFootGRF_Z"]
                rightFootGRF = element["rightFootGRF_Z"]

                if((phaseAngle > 1550) and (rightFootGRF > 0)):
                    endIndex = count
                    break 

            #print(len(cassie_pose_array[i,0]))
            cassie_pose_array[i,0] = cassie_pose_array[i,0][0:endIndex]
            #print(len(cassie_pose_array[i,0]))


        np.savez('cassiePoses', x=cassie_pose_array)
    
    return np.mean(ep_returns)

class EvalProcessClass():
    def __init__(self, args, run_args):
        self.args = args
        self.run_args = run_args
        # if len([x for x in [args.gait, args.pca, args.pds] if x is True]) > 1:
        #     print("right now multiple live plots does not seem to work.")
        #     exit(1)

    # def create_live_plots(self, policy, env):
    #     if self.args.gait:
    #         from .gait_plot import Live_Gait_Plot
    #         self.gait_plot_pipe, gait_plotter_pipe = mp.Pipe()
    #         gait_plotter = Live_Gait_Plot(env.behavior)
    #         self.gait_plot_process = mp.Process(target=gait_plotter, args=(gait_plotter_pipe,))
    #         self.gait_plot_process.start()
    #         self.send_gait = self.gait_plot_pipe.send
            
    #     if self.args.pca:
    #         from util.pca import Live_PCA_Plot, get_hiddens
    #         self.pca_plot_pipe, pca_plotter_pipe = mp.Pipe()
    #         pca_plotter = Live_PCA_Plot(policy, env)
    #         self.pca_plot_process = mp.Process(target=pca_plotter, args=(pca_plotter_pipe,))
    #         self.pca_plot_process.start()
    #         self.send_pca = self.pca_plot_pipe.send
    #         self.get_hiddens = get_hiddens
    #     if self.args.pds:
    #         from util.pds import Live_PD_Plot
    #         self.pd_plot_pipe, pd_plotter_pipe = mp.Pipe()
    #         pd_plotter = Live_PD_Plot()
    #         self.pd_plot_process = mp.Process(target=pd_plotter, args=(pd_plotter_pipe,))
    #         self.pd_plot_process.start()
    #         self.send_pd = self.pd_plot_pipe.send
    #         print("DOING PDS??")


    # TODO: Add pausing, and window quiting along with other render functionality
    def eval_policy(self, policy, terrain=None):
        with torch.no_grad():
            m_policy = deepcopy(policy)
            args, run_args = self.args, self.run_args

            if args.debug:
                args.stats = False
            if args.reward is None:
                args.reward = run_args.reward

            def isData():
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

            visualize = not args.no_viz
            print("env name: ", run_args.env)
            if run_args.env is None:
                env_name = args.env
            else:
                env_name = run_args.env

            for key in args.__dict__:
                if key not in run_args:
                    run_args.__dict__[key] = args.__dict__[key]
            run_args.__dict__['terrain'] = args.__dict__['terrain']

            env = env_factory(**vars(run_args))()
            print(f"{env}\n")
            env.dynamics_randomization = False
            env.evaluation_mode = True
            env.debug = args.debug
            state = env.reset()
            # self.create_live_plots(policy, env)

            # if self.args.pca:
            #     from util.pca import PCA_Plot
            #     pca_plot = PCA_Plot(policy, env)

            # if self.args.pds:
            #     from util.pds import PD_Plot
            #     pd_plot = PD_Plot(policy, env)
            #     print("DOING PDS??")

            # if self.args.gait:
            #     from .gait_plot import Gait_Plot
            #     gait_plot = Gait_Plot(env.behavior)

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()
                m_policy.init_hidden_state()

            old_settings = termios.tcgetattr(sys.stdin)

            # Command inputs
            mirror = False
            slowmo = False
            recording = False

            #if visualize:
            env.render()
            render_state = True
            try:
                tty.setcbreak(sys.stdin.fileno())

                state = env.reset()

                env.speed = 0
                env.side_speed = 0
                env.phase_add = 50
                if hasattr(env, 'behavior'):
                    period_shift = env.behavior.period_shift
                    ratio = env.behavior.ratio
                else:
                    period_shift = env.period_shift
                    ratio = env.ratio
                done = False
                timesteps = 0
                eval_reward = 0

                while render_state:
                
                    if isData():
                        clock_change = False
                        c = sys.stdin.read(1)

                        if c == 'r':
                            state = env.reset()
                            if hasattr(policy, 'init_hidden_state'):
                                policy.init_hidden_state()
                                m_policy.init_hidden_state()
                            print("Resetting environment via env.reset()")
                        elif c == 'x':
                            if hasattr(policy, 'init_hidden_state'):
                                policy.init_hidden_state()
                                m_policy.init_hidden_state()
                        
                        elif c == 'v':
                            if not recording:
                                print("Starting Recording!")
                                recording = True
                                recording_name = os.path.basename(os.path.normpath(args.path)) + datetime.now().strftime("%H_%M_%S") + ".mp4"
                                env.vis.start_video_recording(recording_name, 1920, 1080)
                            else:
                                print("Stopping Recording!")
                                recording = False
                                env.vis.close_video_recording()

                        elif c == 'm':
                            mirror = not mirror
                        else:
                            env.interactive_control(c)

                    start = time.time()

                    if (not env.vis.ispaused()):

                        action   = policy(torch.Tensor(state)).numpy()
                        if mirror and hasattr(env, 'mirror_state') and hasattr(env, 'mirror_action'):
                            m_action = env.mirror_action(m_policy(torch.Tensor(env.mirror_state(state))).numpy())

                        if mirror:
                            action = m_action

                        state, reward, done, _ = env.step(action)
                        eval_reward += reward
                        timesteps += 1
                        qvel = env.sim.qvel()
                        actual_speed = np.linalg.norm(qvel[0:2])
                        if self.args.pca:
                            # pca_plot.update(policy)
                            self.send_pca(self.get_hiddens(policy))
                        if self.args.pds:
                            # pd_plot.update(action, env.l_foot_frc, env.r_foot_frc)
                            self.send_pd((action, env.l_foot_frc, env.r_foot_frc))
                        if self.args.gait:
                            # gait_plot.update(env.behavior)
                            self.send_gait(env.behavior)
                        # print("actual speed: ", np.linalg.norm(qvel[0:2]))
                        # print("commanded speed: ", env.speed)

                        #if args.no_viz:
                        #    yaw = quaternion2euler(new_orient)[2]
                        #    print("stp = {}  yaw = {:.2f}  spd = {}  ep_r = {:.2f}  stp_r = {:.2f}".format(timesteps, yaw, speed, eval_reward, reward))

                        if True:
                            os.system('clear')
                            print(env.log_cmd())
                            #print(float(state[-1:]))
                        #time.sleep(0.1)

                        if recording:
                            print("recording!")
                            env.vis.record_frame()

                            # print("Mirror: {} | Des. Spd. {:5.2f} | Speed {:5.1f} | Sidespeed {:4.2f} | Heading {:5.2f} | Freq. {:3d} | Ratio {:3.2f},{:3.2f} | RShift {:3.2f},{:3.2f} | {:20s}".format(mirror, env.speed, actual_speed, env.side_speed, env.orient_add, int(env.phase_add), *ratio, *period_shift, ''), end='\r')

                    #if visualize:
                    #    render_state = env.render()
                    
                    #    env.precompute_clock()
                    render_state = env.render()
                    if hasattr(env, 'simrate'):
                        # assume 40hz
                        end = time.time()
                        delaytime = max(0, 1000 / 40000 - (end-start))
                        if slowmo:
                            while(time.time() - end < delaytime*10):
                                env.render()
                                time.sleep(delaytime)

                        else:
                            time.sleep(delaytime)
                    else:
                        time.sleep(0.02)


                print("Eval reward: ", eval_reward)

            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                time.sleep(0.02)
                if self.args.pca:
                    self.pca_plot_process.join()
                if self.args.pds:
                    self.pd_plot_process.join()
                if self.args.gait:
                    self.gait_plot_process.join()


class EvalProcess_GestureCommands(EvalProcessClass):
    def __init__(self, args, run_args):
        super().__init__(args, run_args)
        #sys.path.append("../../gesture-recognition-robot")
        #sys.path.append('~/ROB514/finalProj/gesture-recognition-robot/')
        sys.path.append( '../gesture-recognition-robot' )
        
        from run_gesture_recognition import ProcessGesture
        import queue
        import threading

        self.q = queue.Queue()
        self.myGesture = ProcessGesture()
        t1 = threading.Thread(target=self.myGesture.run_gesture_recognition, args= (self.q,))
        t1.start()
 
        self.moveDict = {

            "thumbs up" : "forward",
            "call me" : "forward",
            "thumbs down" : "backward",
            "fist" : "stop",
            "rock" : "leftTurn",
            "peace" : "rightTurn"
        
        }

        #from run_gesture_recognition import ProcessGesture
        
        #self.lastCommand = "stop"

        # if len([x for x in [args.gait, args.pca, args.pds] if x is True]) > 1:
        #     print("right now multiple live plots does not seem to work.")
        #     exit(1)

    # def create_live_plots(self, policy, env):
    #     if self.args.gait:
    #         from .gait_plot import Live_Gait_Plot
    #         self.gait_plot_pipe, gait_plotter_pipe = mp.Pipe()
    #         gait_plotter = Live_Gait_Plot(env.behavior)
    #         self.gait_plot_process = mp.Process(target=gait_plotter, args=(gait_plotter_pipe,))
    #         self.gait_plot_process.start()
    #         self.send_gait = self.gait_plot_pipe.send
            
    #     if self.args.pca:
    #         from util.pca import Live_PCA_Plot, get_hiddens
    #         self.pca_plot_pipe, pca_plotter_pipe = mp.Pipe()
    #         pca_plotter = Live_PCA_Plot(policy, env)
    #         self.pca_plot_process = mp.Process(target=pca_plotter, args=(pca_plotter_pipe,))
    #         self.pca_plot_process.start()
    #         self.send_pca = self.pca_plot_pipe.send
    #         self.get_hiddens = get_hiddens
    #     if self.args.pds:
    #         from util.pds import Live_PD_Plot
    #         self.pd_plot_pipe, pd_plotter_pipe = mp.Pipe()
    #         pd_plotter = Live_PD_Plot()
    #         self.pd_plot_process = mp.Process(target=pd_plotter, args=(pd_plotter_pipe,))
    #         self.pd_plot_process.start()
    #         self.send_pd = self.pd_plot_pipe.send
    #         print("DOING PDS??")

    

    def getGestureCommand(self):
        """
        
        placeholder for gesture recognition 

        """

        #"type forward, backward, leftStep, rightStep, leftTurn, rightTurn, or stop"

        # import random

        # diceRoll = random.randrange(0,100)

        moveList = ["forward", "backward", "leftStep", "rightStep", "leftTurn", "rightTurn","stop"]

        # if(diceRoll == 0):

        #     nextCommand = random.choice(moveList)
        #     self.lastCommand = nextCommand

        # return self.lastCommand

        #value = self.q.get()

        gestureCommand = self.myGesture.current_command
        if gestureCommand in self.moveDict:
            return self.moveDict[gestureCommand]
        else:
            return self.moveDict["fist"]


    # TODO: Add pausing, and window quiting along with other render functionality
    def eval_policy(self, policy, terrain=None):
        with torch.no_grad():
            m_policy = deepcopy(policy)
            args, run_args = self.args, self.run_args

            if args.debug:
                args.stats = False
            if args.reward is None:
                args.reward = run_args.reward

            def isData():
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

            visualize = not args.no_viz
            print("env name: ", run_args.env)
            if run_args.env is None:
                env_name = args.env
            else:
                env_name = run_args.env

            for key in args.__dict__:
                if key not in run_args:
                    run_args.__dict__[key] = args.__dict__[key]
            run_args.__dict__['terrain'] = args.__dict__['terrain']

            env = env_factory(**vars(run_args))()
            print(f"{env}\n")
            env.dynamics_randomization = False
            env.evaluation_mode = True
            env.debug = args.debug
            state = env.reset()
            # self.create_live_plots(policy, env)

            # if self.args.pca:
            #     from util.pca import PCA_Plot
            #     pca_plot = PCA_Plot(policy, env)

            # if self.args.pds:
            #     from util.pds import PD_Plot
            #     pd_plot = PD_Plot(policy, env)
            #     print("DOING PDS??")

            # if self.args.gait:
            #     from .gait_plot import Gait_Plot
            #     gait_plot = Gait_Plot(env.behavior)

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()
                m_policy.init_hidden_state()

            old_settings = termios.tcgetattr(sys.stdin)

            # Command inputs
            mirror = False
            slowmo = False
            recording = False

            #if visualize:
            env.render()
            render_state = True
            try:
                tty.setcbreak(sys.stdin.fileno())

                state = env.reset()

                env.speed = 0
                env.side_speed = 0
                env.phase_add = 50
                if hasattr(env, 'behavior'):
                    period_shift = env.behavior.period_shift
                    ratio = env.behavior.ratio
                else:
                    period_shift = env.period_shift
                    ratio = env.ratio
                done = False
                timesteps = 0
                eval_reward = 0

                while render_state:
                    
                    #if isData():
                    clock_change = False
                    #c = sys.stdin.read(1)

                    #c = input("type forward, backward, leftStep, rightStep, leftTurn, rightTurn, or stop")
                    forwardVel = 0.5
                    backwardVel = 0.2
                    c = self.getGestureCommand()
                    print("commanded action: {nextCommand}".format(nextCommand = c))

                    if c == 'r':
                        state = env.reset()
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                            m_policy.init_hidden_state()
                        print("Resetting environment via env.reset()")
                    elif c == 'x':
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                            m_policy.init_hidden_state()
                    
                    elif c == 'v':
                        if not recording:
                            print("Starting Recording!")
                            recording = True
                            recording_name = os.path.basename(os.path.normpath(args.path)) + datetime.now().strftime("%H_%M_%S") + ".mp4"
                            env.vis.start_video_recording(recording_name, 1920, 1080)
                        else:
                            print("Stopping Recording!")
                            recording = False
                            env.vis.close_video_recording()

                    elif c == 'm':
                        mirror = not mirror
                    else:
                        env.interactive_control_gesture_cmd(c, forwardVel, backwardVel)

                    start = time.time()

                    if (not env.vis.ispaused()):

                        action   = policy(torch.Tensor(state)).numpy()
                        if mirror and hasattr(env, 'mirror_state') and hasattr(env, 'mirror_action'):
                            m_action = env.mirror_action(m_policy(torch.Tensor(env.mirror_state(state))).numpy())

                        if mirror:
                            action = m_action

                        state, reward, done, _ = env.step(action)
                        eval_reward += reward
                        timesteps += 1
                        qvel = env.sim.qvel()
                        actual_speed = np.linalg.norm(qvel[0:2])
                        if self.args.pca:
                            # pca_plot.update(policy)
                            self.send_pca(self.get_hiddens(policy))
                        if self.args.pds:
                            # pd_plot.update(action, env.l_foot_frc, env.r_foot_frc)
                            self.send_pd((action, env.l_foot_frc, env.r_foot_frc))
                        if self.args.gait:
                            # gait_plot.update(env.behavior)
                            self.send_gait(env.behavior)
                        # print("actual speed: ", np.linalg.norm(qvel[0:2]))
                        # print("commanded speed: ", env.speed)

                        #if args.no_viz:
                        #    yaw = quaternion2euler(new_orient)[2]
                        #    print("stp = {}  yaw = {:.2f}  spd = {}  ep_r = {:.2f}  stp_r = {:.2f}".format(timesteps, yaw, speed, eval_reward, reward))

                        if True:
                            os.system('clear')
                            print(env.log_cmd())
                            #print(float(state[-1:]))
                        #time.sleep(0.1)

                        if recording:
                            print("recording!")
                            env.vis.record_frame()

                            # print("Mirror: {} | Des. Spd. {:5.2f} | Speed {:5.1f} | Sidespeed {:4.2f} | Heading {:5.2f} | Freq. {:3d} | Ratio {:3.2f},{:3.2f} | RShift {:3.2f},{:3.2f} | {:20s}".format(mirror, env.speed, actual_speed, env.side_speed, env.orient_add, int(env.phase_add), *ratio, *period_shift, ''), end='\r')

                    #if visualize:
                    #    render_state = env.render()
                    
                    #    env.precompute_clock()
                    render_state = env.render()
                    if hasattr(env, 'simrate'):
                        # assume 40hz
                        end = time.time()
                        delaytime = max(0, 1000 / 40000 - (end-start))
                        if slowmo:
                            while(time.time() - end < delaytime*10):
                                env.render()
                                time.sleep(delaytime)

                        else:
                            time.sleep(delaytime)
                    else:
                        time.sleep(0.02)


                print("Eval reward: ", eval_reward)

            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                time.sleep(0.02)
                if self.args.pca:
                    self.pca_plot_process.join()
                if self.args.pds:
                    self.pd_plot_process.join()
                if self.args.gait:
                    self.gait_plot_process.join()

