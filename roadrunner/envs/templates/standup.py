import numpy as np
import random

from envs.terrains.stairs import Stairs

from util.misc import remap
from util.quat import *

class StandupEnv:
    def __init__(self, simrate=50,
                       dynamics_randomization=False,
                       terrain=False,
                       **kwargs):

        self.super_interface = set()
        obs_size = self.state_est_size #+ 1

        self.observation_space = np.zeros(obs_size)
        self.targets = self.offset

        from envs.rewards.standup_reward import setup_reward_components, compute_reward, compute_done
        self.w = setup_reward_components(self)
        self._compute_reward = compute_reward
        self._compute_done = compute_done

        self.action_space = np.zeros(self.num_actuators)

    def reset_info(self):
        self.time               = 0
        self.last_action        = None
        self.avg_ctrl_penalty   = 0
        self.avg_trq_penalty    = 0
        self.torque_sum         = 0
        self.orient_add         = 0

        self.soft_countdown = np.random.randint(200, 250)
        #self.hard_countdown = np.random.randint(self.soft_countdown+10, 300)
        #self.hard_cutoff = np.random.randint(250, 400)

    def get_soft_countdown(self):
        if self.time < self.soft_countdown:
            return remap(self.time, 0, self.soft_countdown, 0, 1)
        else:
            return 1

    """
    def get_hard_countdown(self):
        return self.get_soft_countdown()
        if self.time < self.soft_countdown:
            return 0
        elif self.time >= self.soft_countdown and self.time <= self.hard_countdown:
            return remap(self.time, self.soft_countdown, self.hard_countdown, 0, 1)
        else:
            return 1
    """

    def get_full_state(self):
        ext_state = []
        robot_state = self.get_info('robot_state')

        return np.concatenate([robot_state, ext_state])

    def step(self, action, *args, **kwargs):

        delay_rand = 3
        if self.dynamics_randomization:
            self.simrate = self.default_simrate + np.random.randint(-delay_rand, delay_rand+1)
        else:
            self.simrate = self.default_simrate

        """ Run the simulation forward by self.simrate steps. This call
            will update any tracker functions in self.tracker_fn.
        """
        self.targets = self.get_info('robot_motor_positions')
        self.simulation_forward(action, n=self.simrate)
        r = self.compute_reward(action)
        self.time += 1

        _, _, self.orient_add = quaternion2euler(self.get_info('robot_orientation', rotate_to_heading=False))
        pitch, roll, yaw = quaternion2euler(self.get_info('robot_orientation', rotate_to_heading=True))

        # Action cost term
        if self.last_action is None:
            ctrl_penalty = 0
        else:
            ctrl_penalty = 5 * sum(np.abs(self.last_action - action)) / len(action)

        torque = self.get_info('smoothed_torques')
        torque_penalty = 0.05 * sum(np.abs(torque)/len(torque))
        self.avg_ctrl_penalty = self.avg_ctrl_penalty * 0.95 + ctrl_penalty * 0.05
        self.avg_trq_penalty  = self.avg_trq_penalty * 0.95 + torque_penalty * 0.05

        return self.get_full_state(), r, self.compute_done(), {}

    def compute_done(self):
        height_penalty = 3 * self.get_soft_countdown() * np.abs(self.get_info('robot_height', naive=True) - 0.8)
        if np.exp(-height_penalty) < 0.8:
            return True
        return False

    def _compute_orientation_penalty(self):
        actual_q = self.get_info('robot_orientation', rotate_to_heading=True)
        target_q = [1, 0, 0, 0]
        orientation_error = (1 - np.inner(actual_q, target_q) ** 2)

        #left_actual, right_actual = self.get_info('robot_foot_orientations', local=False)
        #left_actual_target_euler  = quaternion2euler(left_actual)  * [0, 1, 0] # ROLL PITCH YAW
        #right_actual_target_euler = quaternion2euler(right_actual) * [0, 1, 0]
        #left_actual_target        = euler2quat(z=left_actual_target_euler[2], y=left_actual_target_euler[1], x=left_actual_target_euler[0])
        #right_actual_target       = euler2quat(z=right_actual_target_euler[2], y=right_actual_target_euler[1], x=right_actual_target_euler[0])
        #foot_err = 2 * ((1 - np.inner(left_actual, left_actual_target) ** 2) + (1 - np.inner(right_actual, right_actual_target) ** 2))

        return orientation_error# + foot_err

    def compute_reward(self, action):
        orientation_penalty = self._compute_orientation_penalty()

        height_penalty = self.get_soft_countdown() * 3 * np.abs(self.get_info('robot_height', naive=True) - 0.8)
        reward = 0.00 + \
                 0.10 * np.exp(-self.avg_ctrl_penalty) + \
                 0.15 * np.exp(-orientation_penalty) + \
                 0.30 * np.exp(-height_penalty) + \
                 0.45 * np.exp(-self.avg_trq_penalty)
        return reward

    def get_action_mirror_indices(self):
        mirror_act = [x for x in self.get_info('action_mirror_indices')]
        return mirror_act

    def get_state_mirror_indices(self):
        mirror_obs = [x for x in self.get_info('state_mirror_indices')]
        #mirror_obs += [len(mirror_obs)]
        return mirror_obs
