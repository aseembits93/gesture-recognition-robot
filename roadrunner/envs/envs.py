from envs.cassie.cassie import CassieEnv
from envs.templates.locomotion import LocomotionEnv
from envs.templates.standup import StandupEnv

from envs.cassie.cassiemujoco import pd_in_t
from util.quat import *

import numpy as np
import json 
import random

class CassieLocomotionEnv(CassieEnv, LocomotionEnv):
    def __init__(self, **kwargs):
        CassieEnv.__init__(self, **kwargs)
        LocomotionEnv.__init__(self,**kwargs)

class CassieFFWalkEnv(CassieEnv, LocomotionEnv):
    def __init__(self, **kwargs):
        CassieLocomotionEnv.__init__(self, **kwargs)
        self.state_est_size  = 44
        self.robot_info_registry['robot_state_size']      = lambda: self.state_est_size
        self.observation_size = 44 + 2 + 1 # state_est_size + clock_size(2) + xvel command(1)
        self.observation_space = np.zeros(self.observation_size)

        self.robot_state_mirror_indices = [0.01, -1, 2, -3,      # pelvis orientation
                                          
                                          -4, 5, -6,             # rotational vel

                                           7, -8, 9,              #pelvis vel

                                           13, -14, 15,           #left foot pos

                                           10, -11, 12,           #right foot pos

                                          -21, -22, 23, 24, 25,  # left motor pos
                                          
                                          -16,  -17,  18,  19,  20, # right motor pos

                                          -31, -32, 33, 34, 35,  # left motor vel
                                          
                                          -26, -27, 28, 29, 30,  # right motor vel

                                          38, 39, 36, 37,        # joint pos
                                          
                                          42, 43, 40, 41, ]      # joint vel
        
        # Changing gait parameter bounds to just include walking/running. Though step_freq and ratio bounds
        # are not used in this case, see randomize_super_command function.
        self.speed_bounds      = [0.0, 4.0]
        self.side_speed_bounds = [0.0, 0.0]
        self.step_freq_bounds  = [1.0, 1.5]
        self.ratio_bounds      = [0.4, 0.8]

        # Adjust the phaselen (cycle time). phaselen is in number of 2kHz steps, so the default
        # phaselen of 1700 is equivalent to 1700*0.0005 = 0.85 seconds.
        self.phaselen = 1600    # equivalent to 0.8 seconds

        # Some flags for what type of training
        # Change the commanded speed during a rollout. Not neccessary, but can increase robustness
        self.train_speedchange = False
        # Set turn commands during a rollout. Not neccessary, quaternion randomization at init is enough, but can increase robustness
        self.train_turn = False 
        # print("state est size:", self.get_info('robot_state_size'))

        npzfile = np.load('./cassiePoses.npz', allow_pickle=True)
        self.poseArray = npzfile['x']

        
    def get_full_state(self):
        clock = self.input_clocks()

        # Since just doing walking/running, don't need to input y_vel since it is not changing and swing ratio 
        # is directly linked to x_vel command (see randomize_super_command function). turn_rate is not part of input
        # either, just uses old method of changing inputted quaternion to control heading.
        ext_state = [self.cmd_dict['x_vel']]
    
        ext_state   = np.hstack((clock, ext_state))
        robot_state = self.get_info('robot_state')

        return np.concatenate([robot_state, ext_state])    

    def get_robot_state(self):
       ### include estimated robot pelvis velocity in robot_state input 
        if self.dynamics_randomization:
            motor_pos = self.robot_state.motor.position[:] + self.motor_encoder_noise
            joint_pos = self.robot_state.joint.position[:] + self.joint_encoder_noise
        else:
            motor_pos = self.robot_state.motor.position[:]
            joint_pos = self.robot_state.joint.position[:]

        motor_vel = self.robot_state.motor.velocity[:]
        joint_vel = self.robot_state.joint.velocity[:]

        # remove double-counted joint/motor positions
        joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
        joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])

        robot_state = np.concatenate([

            self.rotate_to_heading(self.robot_state.pelvis.orientation), # pelvis orientation
            self.robot_state.pelvis.rotationalVelocity[:],               # pelvis rotational velocity
            self.rotate_to_heading(self.robot_state.pelvis.translationalVelocity[:]),  # pelvis translational velocity
            self.robot_state.leftFoot.position[:],                       # Left foot position
            self.robot_state.rightFoot.position[:],                      # Right foot position
            motor_pos,                                                   # actuated joint positions
            motor_vel,                                                   # actuated joint velocities
            joint_pos,                                                   # unactuated joint positions
            joint_vel                                                    # unactuated joint velocities
        ])

        return robot_state

    def get_state_mirror_indices(self):
        mirror_obs = [x for x in self.get_info('state_mirror_indices')]
        # Note that in this case, the clock is just a clock signal so don't just swap the clock values,
        # need to shift them half a period forward. See mirror.py in util
        mirror_obs += [-len(mirror_obs), -(len(mirror_obs) + 1)] # clock inputs
        mirror_obs += [len(mirror_obs)] # xvel

        return mirror_obs

    def randomize_super_command(self, init=False):
        """
            For randomizing command when higher-level planner is not there.
        """
        clock_changed = False

        cmd = {}
        if init:
            # To better stepping in place performance (decrease drift) have 1/4 chance to just walk in place
            if random.randint(0, 4) == 0:
                cmd['x_vel'] = 0
            else:
                cmd['x_vel'] = np.random.uniform(*self.get_info('speed_bounds'))
        elif np.random.randint(300) == 0 and False:  # random changes to speed
            cmd['x_vel'] = np.random.uniform(*self.get_info('speed_bounds'))
        else:
            cmd['x_vel'] = self.cmd_dict['x_vel']

        if init:
            cmd['y_vel'] = 0
        elif np.random.randint(300) == 0:  # random changes to sidespeed
            cmd['y_vel'] = np.random.uniform(*self.get_info('side_speed_bounds'))
        else:
            cmd['y_vel'] = self.cmd_dict['y_vel']

        if cmd['x_vel'] > 1:
            new_ratio = 0.4 + .4*(cmd['x_vel'] - 1)/3
            cmd['ratio'] = [new_ratio, 1-new_ratio]
            cmd['step_freq'] = 1 + 0.5*(cmd['x_vel'] - 1)/2
        elif cmd['x_vel'] > 3:
            cmd['ratio'] = [0.8, 0.2]
            cmd['step_freq'] = 1.5
        else:
            cmd['ratio'] = [.4, 0.6]
            cmd['step_freq'] = 1.0
        cmd['period_shift'] = [0, 0.5]

        if np.random.randint(200) == 0 or init:
            cmd['turn_rate'] = 0
        elif np.random.randint(200) == 0 and False:  # random changes to orientation
            cmd['turn_rate'] = np.random.uniform(-0.005 * np.pi, 0.005 * np.pi)
        else:
            cmd['turn_rate'] = self.cmd_dict['turn_rate']
            


        cmd['gaze_yaw']   = 0 
        cmd['gaze_pitch'] = 0 

        return cmd

    def input_clocks(self):
        clock = [np.sin(2 * np.pi * ((self.phase/self.phase_len))),
                    np.cos(2 * np.pi * ((self.phase/self.phase_len)))]
        
        return clock

    def requestCassiePose(self):

        """
        Given a gait velocity and a location for the starting point of the gait, return cassie qpos and qvel vectors
        initVel bounds: [0.1,2.0]
        gaitCycle bounds: [0,1]
        """

        initVel = random.randrange(0,20,1)
        #velocityIndex = int(initVel - 1) #formula to convert desired commanded velocity to its corresponding array index 
        
        #if left foot first
        poseElement = random.choice(self.poseArray[initVel][0])
        #if right foot first
        #listIndex = int(gaitCyclePhase*listLength) + int(len(self.poseArrayarray[velocityIndex][0])/2)

        qpos = poseElement["qpos"]
        qvel = poseElement["qvel"]
        return qpos.astype(float), qvel.astype(float)

    def simulation_reset(self, **kwargs):
        # Randomize dynamics:
        if self.dynamics_randomization:
            self.randomize_dynamics()
        else:
            self.default_dynamics()
        
        # Position and velocity randomization. Note that normal CassieLocomotionEnv does not do this, it always resets to the same position,
        # which is fine due to chance to randomization command at every step. Initialization randomization can help increase robustness though.
        # Pick out random qpos and qvel from reset states
        qpos, qvel = self.requestCassiePose()
        # Do additional randomization on x and y pelvis velocity as well as quaternion
        x_size = 0.3
        y_size = 0.2
        qvel[0] += np.random.random() * 2 * x_size - x_size
        qvel[1] = np.random.random() * 2 * y_size - y_size
        orientation = random.randint(-10, 10) * np.pi / 25
        quaternion = euler2quat(z=orientation, y=0, x=0)
        qpos[3:7] = quaternion

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # TODO: Still need to come up with better way to update state est instead of just calling step_pd with empty control
        self.u = pd_in_t()
        self.robot_state = self.sim.step_pd(self.u)
    
    def reset(self):
        """
            Reset for new episode.
        """
        self.reset_info()
        self.simulation_reset()
        self.cmd_dict = self.randomize_super_command(init=True)
        self.phase = np.random.randint(0, self.phase_len)
        self.phase_add = int((self.default_simrate) * self.cmd_dict['step_freq'])

        return self.get_full_state()

class Cassie3StepEnv(CassieEnv, LocomotionEnv):
    def __init__(self, **kwargs):
        CassieEnv.__init__(self, **kwargs)
        LocomotionEnv.__init__(self, **kwargs)
        self.speed_bounds      = [-0.2, 2.0]
        self.side_speed_bounds = [-0.2, 0.2]
        self.step_freq_bounds  = [0.9, 1.3]
        self.height_bounds     = [0.7, 1.0]
        self.ratio_bounds      = [0.35, 0.70]
        self.robot_state_mirror_indices = [0.01, -1, 2, -3,      # pelvis orientation
                                          -4, 5, -6,             # rotational vel
                                          -12, -13, 14, 15, 16,  # left motor pos
                                          -7,  -8,  9,  10,  11, # right motor pos
                                          -22, -23, 24, 25, 26,  # left motor vel
                                          -17, -18, 19, 20, 21,  # right motor vel 
                                          29, 30, 27, 28,        # joint pos
                                          33, 34, 31, 32, ]      # joint vel

        jsonFilePath = "./turning_trajectory_library.json"
        npzfile = np.load('./cassiePoses.npz', allow_pickle=True)
        self.poseArray = npzfile['x']

        with open(jsonFilePath) as trajOptData:
            self.trajData = json.load(trajOptData)
        self.initVel = None

    def initStartingGaitVel(self):

        #randomly choose b/t [1,21) to select gait velocity
        initVel = random.randrange(1,21,1)

        footGRFs = self.trajData[initVel]["input"][2]

        numOfNodesInAerial = 0
        if(footGRFs[0] == 0):
            for element in footGRFs:
                if(element == 0):
                    numOfNodesInAerial += 1
                else:
                    break

        gaitCycleFraction = 1 - (numOfNodesInAerial/50)

        return initVel, gaitCycleFraction

    def requestCassiePose(self, initVel, gaitCyclePhase):

        """
        Given a gait velocity and a location for the starting point of the gait, return cassie qpos and qvel vectors
        initVel bounds: [0.1,2.0]
        gaitCycle bounds: [0,1]
        """

        velocityIndex = int(initVel - 1) #formula to convert desired commanded velocity to its corresponding array index 

        listLength = (len(self.poseArray[velocityIndex][0])/2)-90
        
        #if left foot first
        listIndex = int(gaitCyclePhase*listLength)
        #if right foot first
        #listIndex = int(gaitCyclePhase*listLength) + int(len(self.poseArrayarray[velocityIndex][0])/2)

        qpos = self.poseArrayarray[velocityIndex][0][listIndex]["qpos"]
        qvel = self.poseArrayarray[velocityIndex][0][listIndex]["qvel"]
        return qpos.astype(float), qvel.astype(float)

    def simulation_reset(self):

            # Randomize dynamics:
        if self.dynamics_randomization:
            self.randomize_dynamics()
        else:
            self.default_dynamics()
        
        #randomly select an initial velocity, and set qpos and qvel accordingly

        initialVel, gaitCyclePosition = self.initStartingGaitVel()

        qpos, qvel = self.requestCassiePose(initialVel, gaitCyclePosition)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        self.robot_state = self.sim.step_pd(self.u)

    def reset(self):
        """
            Reset for new episode.
        """
        self.reset_info()
        self.simulation_reset()
        self.cmd_dict = self.randomize_super_command(init=True)
        return self.get_full_state()

class CassieStandupEnv(CassieEnv, StandupEnv):
    def __init__(self, **kwargs):
        CassieEnv.__init__(self, **kwargs)
        StandupEnv.__init__(self, **kwargs)

    def reset(self):
        self.reset_info()
        self.simulation_reset()

        for _ in range(20):
            act = np.random.randn(*self.action_space.shape) * 10
            self.simulation_forward(act, n=50)

        for _ in range(50):
            act = np.zeros(30)
            act[10:20] = [-x for x in self.P] + [-x for x in self.P]
            act[20:30] = [-x for x in self.D] + [-x for x in self.D]
            self.simulation_forward(act, n=50)


        _, _, self.orient_add = quaternion2euler(self.get_info('robot_orientation', rotate_to_heading=False))
        return self.get_full_state()

    def step_simulation(self, action):
        self.render()
        target_delta = action[:10]
        p_add  = np.zeros(10)
        d_add  = np.zeros(10)

        if len(action) > 10:
            d_add = action[20:30]

        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i] + p_add[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i] + p_add[i + 5]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i] + d_add[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i] + d_add[i + 5]

            self.u.leftLeg.motorPd.pTarget[i]  = self.targets[i] + target_delta[i]
            self.u.rightLeg.motorPd.pTarget[i] = self.targets[i + 5] + target_delta[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

        self.robot_state = self.sim.step_pd(self.u)

