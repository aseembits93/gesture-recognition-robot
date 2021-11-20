import numpy as np
import random
import os

from envs.digit.digitmujoco import DigitSim, DigitVis, pd_in_t, digit_out_t
from envs.pd_env import PD_Env
from util.quat import *

class DigitEnv(PD_Env):
    def __init__(self, *args, **kwargs):
        self.perception_size = 6
        self.state_est_size  = 55
        self.num_actuators   = 20

        self.sim = DigitSim(terrain=kwargs['terrain'], perception=kwargs['perception'])

        self.P = np.array([80,  120,  300,  250,  50,  50,
                           80,  120,  300,  250,  50,  50,
                           50,    50,  50,  50,
                           50,    50,  50,  50])
        self.D = np.array([5,  5,  10,  10,  5,  5,
                           5,  5,  10,  10,  5,  5,
                           5,    5,  5,  5,
                           5,    5,  5,  5])
        self.u = pd_in_t()
        self.robot_state = None

        self.simrate_bounds = [0.95, 1.05]
        self.incline_bounds = [-0.03, 0.03]
        self.encoder_noise  = 0.0005
        self.damping_bounds = [0.5, 3.5]
        self.mass_bounds    = [0.5, 1.7]
        self.fric_bounds    = [0.35, 1.1]

        # Record default dynamics parameters
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass    = self.sim.get_body_mass()
        self.default_ipos    = self.sim.get_body_ipos()
        self.default_fric    = self.sim.get_geom_friction()
        self.default_rgba    = self.sim.get_geom_rgba()
        self.default_quat    = self.sim.get_geom_quat()

        self.motor_encoder_noise = np.zeros(20)
        self.joint_encoder_noise = np.zeros(4)

        self.state_est_indices = [0.001, 1, 2, 3,                      # pelvis orientation
                                         -4, 5, -6,                    # rotational vel
                                         -13, -14, -15, -16, -17, -18, # left leg motor pos
                                         -7,  -8,  -9,  -10, -11, -12, # right leg motor pos
                                         -23, -24, -25, -26,           # left arm motor pos
                                         -19, -20, -21, -22,           # right arm motor pos
                                         -33, -34, -35, -36, -37, -38, # left left motor vel
                                         -27, -28, -29, -30, -31, -32, # right leg motor vel
                                         -43, -44, -45, -46,           # left arm motor vel
                                         -39, -40, -41, -42,           # right arm motor vel
                                         -49, -50, -47, -48,           # joint pos
                                         -53, -54, -51, -52]           # joint vel

        PD_Env.__init__(**kwargs)

        # Adding to the registry
        
        self.robot_const_registry['simrate_bounds'] = self.simrate_bounds
        self.robot_const_registry['incline_bounds'] = self.incline_bounds
        self.robot_const_registry['encoder_noise'] = self.encoder_noise
        self.robot_const_registry['damping_bounds'] = self.damping_bounds
        self.robot_const_registry['mass_bounds'] = self.mass_bounds

        self.robot_const_registry['speed_bounds'] = self.speed_bounds
        self.robot_const_registry['side_speed_bounds'] = self.side_speed_bounds
        self.robot_const_registry['step_freq_bounds'] = self.step_freq_bounds
        self.robot_const_registry['gaze_bounds'] = self.gaze_bounds
        self.robot_const_registry['height_bounds'] = self.height_bounds
        self.robot_const_registry['ratio_bounds'] = self.ratio_bounds

    def step_simulation(self, action):

        target = action[:20]
        p_add  = np.zeros(20)
        d_add  = np.zeros(20)

        if len(action) > 20:
            p_add = action[20:40]

        if len(action) > 40:
            d_add = action[40:60]

        if self.dynamics_randomization:
            target -= self.motor_encoder_noise

        for i in range(20):
            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            self.u.pGain[i]   = self.P[i] + p_add[i]
            self.u.dGain[i]   = self.D[i] + d_add[i]
            self.u.pTarget[i] = target[i]
            self.u.dTarget[i] = 0
            self.u.torque[i]  = 0

        self.robot_state = self.sim.step_pd(self.u)

    def randomize_dynamics(self):
        fric_noise = []
        translational = np.random.uniform(*self.fric_bounds)
        torsional = np.random.uniform(1e-4, 5e-4)
        rolling = np.random.uniform(1e-4, 2e-4)
        for _ in range(int(len(self.default_fric)/3)):
            fric_noise += [translational, torsional, rolling]

        if not self.stairs:
            geom_plane = [np.random.uniform(*self.incline_bounds), np.random.uniform(*self.incline_bounds), 0]
            quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            geom_quat  = list(quat_plane) + list(self.default_quat[4:])
        else:
            geom_quat = self.default_quat

        self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=20)
        self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=4)

        self.sim.set_body_mass(self.default_mass)
        self.sim.set_body_ipos(self.default_ipos)
        self.sim.set_dof_damping(self.default_damping)
        self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
        self.sim.set_geom_quat(geom_quat)
        self.pd_simrate = int(self.default_simrate)
        self.sim.set_const()

    def default_dynamics(self):
        self.sim.set_body_mass(self.default_mass)
        self.sim.set_body_ipos(self.default_ipos)
        self.sim.set_dof_damping(self.default_damping)
        self.sim.set_geom_friction(self.default_fric)
        self.sim.set_geom_quat(self.default_quat)
        self.pd_simrate = int(self.default_simrate)

        self.motor_encoder_noise = np.zeros(20)
        self.joint_encoder_noise = np.zeros(4)

        self.sim.set_const()

    def render(self):
        if self.vis is None:
            self.vis = DigitVis(self.sim, "./digit/digit-mujoco-sim/model/digit.xml")

        return self.vis.draw(self.sim)

    def get_robot_state(self):

        if self.dynamics_randomization:
            motor_pos = self.robot_state.motor_pos[:] + self.motor_encoder_noise
            joint_pos = self.robot_state.joint_pos[:] + self.joint_encoder_noise
        else:
            motor_pos = self.robot_state.motor_pos[:]
            joint_pos = self.robot_state.joint_pos[:]

        motor_vel = self.robot_state.motor_vel[:]
        joint_vel = self.robot_state.joint_vel[:]

        orientation = self.rotate_to_heading(self.robot_state.torso.vectorNav.orientation[:])
        robot_state = np.concatenate([
            orientation,                                         # pelvis orientation
            self.robot_state.torso.vectorNav.angularVelocity[:], # pelvis rotational velocity 
            motor_pos,                                           # actuated joint positions
            motor_vel,                                           # actuated joint velocities
            joint_pos,                                           # unactuated joint positions
            joint_vel                                            # unactuated joint velocities
        ])

        return robot_state

    """
    labels = [
              'orientation 1',
              'orientation 2',
              'orientation 3',
              'orientation 4',
              'rotation 1',
              'rotation 2',
              'rotation 3',
              '(pos) left hip roll drive',
              '(pos) left hip yaw drive',
              '(pos) left hip pitch drive',
              '(pos) left knee drive',
              '(pos) left toe a drive',
              '(pos) left toe b drive',
              '(pos) right hip roll drive',
              '(pos) right hip yaw drive',
              '(pos) right hip pitch drive',
              '(pos) right knee drive',
              '(pos) right toe a drive',
              '(pos) right toe b drive',
              '(pos) left shoulder roll drive',
              '(pos) left shoulder yaw drive',
              '(pos) left shoulder pitch drive',
              '(pos) left elbow drive',
              '(pos) right shoulder roll drive',
              '(pos) right shoulder yaw drive',
              '(pos) right shoulder pitch drive',
              '(pos) right elbow drive',
              '(vel) left hip roll drive',
              '(vel) left hip yaw drive',
              '(vel) left hip pitch drive',
              '(vel) left knee drive',
              '(vel) left toe a drive',
              '(vel) left toe b drive',
              '(vel) right hip roll drive',
              '(vel) right hip yaw drive',
              '(vel) right hip pitch drive',
              '(vel) right knee drive',
              '(vel) right toe a drive',
              '(vel) right toe b drive',
              '(vel) left shoulder roll drive',
              '(vel) left shoulder yaw drive',
              '(vel) left shoulder pitch drive',
              '(vel) left elbow drive',
              '(vel) right shoulder roll drive',
              '(vel) right shoulder yaw drive',
              '(vel) right shoulder pitch drive',
              '(vel) right elbow drive',
              '(pos) left shin joint',
              '(pos) left tarsus joint',
              '(pos) right shin joint',
              '(vel) right tarsus joint',
              '(vel) left shin joint',
              '(vel) left tarsus joint',
              '(vel) right shin joint',
              '(vel) right tarsus joint']
    """
  
    def generate_left_foot_pos_cmd(self):
        self.last_left_foot, _ = self.get_feet_pos()

        x_target = (self.speed      * (self.pd_simrate / self.phase_add) * 0.85)
        y_target = (self.side_speed * (self.pd_simrate / self.phase_add) * 0.60)
        z_target = 0

        x_target += np.random.uniform(-0.02, 0.02)
        y_target += np.random.uniform(-0.02, 0.02)

        self.left_foot_target = np.array([x_target, y_target, z_target])

    def generate_right_foot_pos_cmd(self):
        _, self.last_right_foot = self.get_feet_pos()
        x_target = (self.speed      * (self.pd_simrate / self.phase_add) * 0.85)
        y_target = (self.side_speed * (self.pd_simrate / self.phase_add) * 0.60)
        z_target = 0

        x_target += np.random.uniform(-0.02, 0.02)
        y_target += np.random.uniform(-0.02, 0.02)

        self.right_foot_target = np.array([x_target, y_target, z_target])
