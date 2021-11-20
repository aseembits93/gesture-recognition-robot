import numpy as np
import random

from envs.templates.periodic import PeriodicEnv

from util.quat import *

from functools import partial

class LocomotionEnv(PeriodicEnv):
    """
        Class used to abstract out commonalities between digit and cassie envs.
    """

    def __init__(self, incentive=False,
                       standing=False,
                       fixed_hop=False,
                       fixed_walk=False,
                       phase_std=0.1,
                       height=False,
                       gaze_control=False,
                       auto_clock=False,
                       impedance=False,
                       debug=False,
                       perception=False,
                       dynamics_randomization=False,
                       reward='icra',
                       antistomp=True,
                       use_stair_heuristic=False,
                       **kwargs):

        self.debug                     = debug
        self.evaluation_mode           = False
        self.incentive                 = incentive
        self.standing                  = standing
        self.hop_only                  = fixed_hop
        self.walk_only                 = fixed_walk
        self.phase_std                 = phase_std
        self.height_commanded          = height
        self.perception                = perception
        self.gaze_control              = gaze_control
        self.auto_clock                = auto_clock
        self.dynamics_randomization    = dynamics_randomization
        self.reward                    = reward
        self.impedance                 = impedance
        self.antistomp                 = antistomp
        self.use_stair_heuristic       = use_stair_heuristic

        if fixed_hop and fixed_walk:
            print("fixed_hop, fixed_walk are mutually exclusive.")
            exit(1)

        obs_size = 0
        self.super_interface = set()
        self.super_interface.add('ratio')
        obs_size += 2

        self.super_interface.add('period_shift')
        obs_size += 2

        self.super_interface.add('turn_rate')  # NOTE: will be included in robot state
        obs_size += 1

        self.super_interface.add('x_vel')
        obs_size += 1

        self.super_interface.add('y_vel')
        obs_size += 1

        if height:
            self.super_interface.add('height')
            obs_size += 1

        if gaze_control:
            self.super_interface.add('gaze_yaw')
            self.super_interface.add('gaze_pitch')
            obs_size += 2

        if not self.auto_clock:
            self.super_interface.add('step_freq')

        if perception:
            obs_size += self.perception_size

        obs_size += self.get_info('robot_state_size')

        self.observation_space = np.zeros(obs_size)

        num_phases = 2
        num_groups = 2
        PeriodicEnv.__init__(self, num_phases,
                                   num_groups,
                                   std=self.phase_std,
                                   phase_len=1700,
                                   phase_add=self.default_simrate)

        if self.reward is None:
            print("WARNING: Defaulting to 'main' reward because no --reward given.")
            self.reward = 'main'

        if self.reward == 'main' or self.reward == 'icra':
            from envs.rewards.icra_locomotion_reward import setup_reward_components, compute_reward, compute_done
            self.w = setup_reward_components(self, incentive=self.incentive)
            self._compute_reward = compute_reward
            self._compute_done = compute_done
        else:
            print("\tNo such reward '{}'.".format(reward))
            raise RuntimeError

        if impedance:
            act_size = self.num_actuators * 3
        else:
            act_size = self.num_actuators

        if self.auto_clock:
            act_size += 1

        self.action_space = np.zeros(act_size)

        # info for extra visuals that will be rendered when evaluating
        self.render_objs = {}

        """ Add the tracker functions we want to run at 2khz to self.trackers. They will
            be called in RobotEnv.simulation_forward().
        """
        self.trackers += [self.energy_tracker, self.distance_tracker, self.peak_grf_tracker]

    def compute_reward(self, action):
        return self._compute_reward(self, action)
    
    def energy_tracker(self, i, final):
        """ Example of a tracker function implemented in a different env. This will be called at 2khz
            inside RobotEnv.simulation_forward().
        """
        curr_motor_pos = np.array(self.get_info('robot_motor_positions', estimate=True)) #@HELEI: change estimate=False to use qpos motor positions if needed

        if not hasattr(self, 'cumulative_energy') or self.time == 0:
            self.cumulative_energy = 0
        else:
            curr_torque    = self.get_info('robot_torque')
            self.cumulative_energy += np.sum(curr_torque * (np.abs(curr_motor_pos - self.prev_motor_pos)))

        if final:
            self.robot_info_registry['cumulative_energy'] = lambda : self.cumulative_energy

        self.prev_motor_pos = curr_motor_pos

    def distance_tracker(self, i, final):
        if not hasattr(self, 'dist_traveled') or self.time == 0:
            self.dist_traveled = 0

        if i == 0:
            self.last_robot_pos = self.get_info('robot_position')

        elif final:
            self.dist_traveled += np.linalg.norm(self.get_info('robot_position') - self.last_robot_pos)
            self.robot_info_registry['distance_traveled'] = lambda : self.dist_traveled
        
    def peak_grf_tracker(self, i, final):
        lfrc, rfrc = self.get_info('robot_foot_forces')

        if not hasattr(self, 'prev_foot_frcs') or self.time == 0:
            self.peak_grf_change = [0, 0]
        else:
            ldelta, rdelta = (x - y for (x,y) in zip((lfrc, rfrc), self.prev_foot_frcs))
            self.peak_grf_change = [max(x,y) for (x,y) in zip(self.peak_grf_change, (ldelta, rdelta))]

        self.prev_foot_frcs = lfrc, rfrc

    def calculate_liftoff_point(self):
        lclock = self.behavior.reward_components['left force']
        rclock = self.behavior.reward_components['right force']
        for i in range(len(lclock)):
            if lclock[i] < -0.5 and lclock[i-1] > -0.5:
                self.liftoff_point[0] = i
            if lclock[i] > -0.5 and lclock[i-1] < -0.5:
                self.touchdown_point[0] = i

        for i in range(len(rclock)):
            if rclock[i] < -0.5 and rclock[i-1] > -0.5:
                self.liftoff_point[1] = i
            if rclock[i] > -0.5 and rclock[i-1] < -0.5:
                self.touchdown_point[1] = i

    def step(self, action, *args, **kwargs):

        self.orient_add += self.cmd_dict['turn_rate']

        delay_rand = 3
        if self.dynamics_randomization:
            self.simrate = self.default_simrate + np.random.randint(-delay_rand, delay_rand+1)
        else:
            self.simrate = self.default_simrate
        

        if self.auto_clock:
            policy_clock_output = action[-1]

            action = action[:-1]

            self.phase_add = policy_clock_output

        """ Run the simulation forward by self.simrate steps. This call
            will update any tracker functions in self.tracker_fn.
        """
        
        if('logging') in kwargs:

            if(kwargs["logging"] == True):

                logData = self.simulation_forward_with_logging(action, n=self.simrate)
        
        else:
            
            self.simulation_forward(action, n=self.simrate)
        
        r = self.compute_reward(action)
        self.update_phase()
        self.time += 1

        # Action cost term
        if self.last_action is None:
            ctrl_penalty = 0
        else:
            ctrl_penalty = 5 * sum(np.abs(self.last_action - action)) / len(action)

        torque = self.get_info('smoothed_torques')
        torque_penalty = 0.05 * sum(np.abs(torque)/len(torque))
        self.avg_ctrl_penalty = self.avg_ctrl_penalty * 0.95 + ctrl_penalty * 0.05
        self.avg_trq_penalty  = self.avg_trq_penalty * 0.95 + torque_penalty * 0.05

        lclock = self.behavior.reward_components['left force'][self.phase]
        rclock = self.behavior.reward_components['right force'][self.phase]
        fpos   = self.get_info('robot_foot_positions')

        self.last_action = action
        stairs_ref = self.stairs

        for i, (clock, pos) in enumerate(zip([lclock, rclock], fpos)):
            if clock > -0.8: # foot in stance
                self.foot_in_stance[i]   = True
                if self.did_liftoff_calc[i] and stairs_ref: # touchdown
                    self.current_stair_step[i], _ = stairs_ref.check_foot_step(*pos)
                self.did_liftoff_calc[i] = False
                self.did_midswing_calc[i] = False

            else:  # foot in swing
                self.foot_in_stance[i] = False

                if stairs_ref and not self.did_liftoff_calc[i] and self.foot_in_stance[(i+1)%2]:  # liftoff and other foot is in stance (for hopping on stairs?)
                    self.last_stair_step[i], _ = stairs_ref.check_foot_step(*pos)
                    if self.current_stair_step[(i+1)%2] is not None:
                        self.next_stair_step[i] = self.current_stair_step[(i+1)%2] + 1
                    else:
                        self.next_stair_step[i] = None

                    self.did_liftoff_calc[i] = True

            if clock < -0.98:
              if not self.did_midswing_calc[i]:
                    self.did_midswing_calc[i] = True
                    if i == 0:
                        self.peak_grf_change = [0, self.peak_grf_change[1]]
                    else:
                        self.peak_grf_change = [self.peak_grf_change[0], 0]
        self.robot_info_registry['peak_foot_forces'] = self.peak_grf_change

        if self.cmd_dict['ratio'] != self.behavior.ratio or self.cmd_dict['period_shift'] != self.behavior.period_shift:
            self.set_behavior(self.cmd_dict['ratio'], self.cmd_dict['period_shift'])
            self.last_clock_change = self.time
            self.calculate_liftoff_point()
            
        #lclock = self.behavior.reward_components['left force'][self.phase]
        #rclock = self.behavior.reward_components['right force'][self.phase]

        if not self.auto_clock:
            if not np.isfinite(self.default_simrate):
              print("NON-FINITE SIMRATE!")
            if not np.isfinite(self.cmd_dict['step_freq']):
              print("NON-FINITE STEP FREQ!", self.cmd_dict['step_freq'])

            self.phase_add = int((self.default_simrate) * self.cmd_dict['step_freq'])

        if not self.evaluation_mode:
            self.cmd_dict = self.randomize_super_command()

        if('logging') in kwargs:

            if(kwargs["logging"] == True):

                return self.get_full_state(), r, self.compute_done(), logData ,{}

        return self.get_full_state(), r, self.compute_done(), {}

    def compute_done(self):
        return self._compute_done(self)

    def randomize_super_command(self, init=False):
        """
            For randomizing command when higher-level planner is not there.
        """
        clock_changed = False

        cmd = {}
        if init:
            cmd['ratio'] = [0.45, 0.55]
            clock_changed = True
        elif np.random.randint(300) == 0:
            cmd['ratio'] = self.randomize_ratio()
            clock_changed = True
        else:
            cmd['ratio'] = self.behavior.ratio

        if init:
            cmd['period_shift'] = [0, 0.5]
            clock_changed = True
        elif np.random.randint(300) == 0:
            cmd['period_shift'] = self.randomize_period_shift()
            clock_changed = True
        else:
            cmd['period_shift'] = self.behavior.period_shift

        if np.random.randint(200) == 0 or init:
            cmd['turn_rate'] = 0
        elif np.random.randint(200) == 0:  # random changes to orientation
            cmd['turn_rate'] = np.random.uniform(-0.01 * np.pi, 0.01 * np.pi)
        else:
            cmd['turn_rate'] = self.cmd_dict['turn_rate']

        if init:
            cmd['x_vel'] = 0
        elif np.random.randint(300) == 0:  # random changes to speed
            cmd['x_vel'] = np.random.uniform(*self.get_info('speed_bounds'))
        else:
            cmd['x_vel'] = self.cmd_dict['x_vel']

        if init:
            cmd['y_vel'] = 0
        elif np.random.randint(300) == 0:  # random changes to sidespeed
            cmd['y_vel'] = np.random.uniform(*self.get_info('side_speed_bounds'))
        else:
            cmd['y_vel'] = self.cmd_dict['y_vel']

        if not self.auto_clock:
            if init:
                cmd['step_freq'] = 1.0
            elif np.random.randint(300) == 0:  # random changes to clock speed
                cmd['step_freq'] = np.random.uniform(*self.get_info('step_freq_bounds'))
            else:
                cmd['step_freq'] = self.cmd_dict['step_freq']

        if self.gaze_control:
            if init:
                cmd['gaze_yaw']   = 0 
                cmd['gaze_pitch'] = 0 
            elif np.random.randint(300) == 0:
                cmd['gaze_yaw']   = np.random.uniform(*self.get_info('gaze_bounds'))
                cmd['gaze_pitch'] = np.random.uniform(*self.get_info('gaze_bounds'))
            else:
                cmd['gaze_yaw']   = self.cmd_dict['gaze_yaw']
                cmd['gaze_pitch'] = self.cmd_dict['gaze_pitch']

        if self.get_info('on_stairs'):
            if init:
                cmd['x_vel'] = np.clip(cmd['x_vel'], 0.6, 0.9)
            else:
                cmd['x_vel'] = np.clip(self.cmd_dict['x_vel'], 0.6, 0.9)

        return cmd

    def get_super_interface(self, **kwargs):
        return self.super_interface

    def reset_info(self):
        """
            Resets the planner local attributes.
        """
        self.time = 0
        self.randomize_phase()
        self.sample_behavior()
        self.phase_add          = int((self.default_simrate) * np.random.uniform(*self.get_info('step_freq_bounds')))
        self.last_robot_pos     = self.get_info('robot_position')

        self.last_action        = None
        self.avg_ctrl_penalty   = 0
        self.avg_trq_penalty    = 0
        self.torque_sum         = 0
        self.orient_add = 0

        self.liftoff_point = [0, 0]
        self.touchdown_point = [0, 0]

        # order is [left, right]
        self.last_stair_step    = [None, None]   # calculated at liftoff
        self.current_stair_step = [None, None]   # calculated at touchdown
        self.next_stair_step    = [None, None]   # calculated at liftoff
        self.did_liftoff_calc   = [False, False]
        self.did_midswing_calc  = [False, False]
        self.foot_in_stance     = [False, False]
        self.foot_caught_edge   = [False, False]

        self.time               = 0
        self.currently_standing = True
        self.calculate_liftoff_point()

    def get_full_state(self):
        clock = self.input_clocks()

        ext_state = [*self.behavior_ratios()]

        ext_state = np.hstack((ext_state, self.cmd_dict['x_vel'], self.cmd_dict['y_vel'], self.cmd_dict['turn_rate']))

        if self.height_commanded:
            ext_state = np.hstack((ext_state, self.cmd_dict['height']))
            
        if self.perception:
            ext_state = np.hstack((ext_state, self.get_info('rangefinder')))

        if self.gaze_control:
            ext_state = np.hstack((ext_state, [self.cmd_dict['gaze_yaw'], self.cmd_dict['gaze_pitch']]))

        ext_state   = np.hstack((ext_state, clock))
        robot_state = self.get_info('robot_state')

        return np.concatenate([robot_state, ext_state])

    def randomize_period_shift(self):
        if self.hop_only:
            return [0, 0]

        if self.walk_only:
            return [0, 0.5]

        self.last_clock_change = self.time
        if np.random.randint(3) == 0:  # 1/3 chance of hopping
            std = 0.03
            if np.random.randint(2) == 0:
                return [np.random.normal(loc=0, scale=std), np.random.normal(loc=0, scale=std)]
            else:
                return [np.random.normal(loc=0.5, scale=std), np.random.normal(loc=0.5, scale=std)]
        else:  # 2/3 chance of walk/running
            std = 0.15
            if np.random.randint(2) == 0:
                return  [np.random.normal(loc=0, scale=std), np.random.normal(loc=0.5, scale=std)]
            else:
                return [np.random.normal(loc=0.5, scale=std), np.random.normal(loc=0, scale=std)]

    def randomize_ratio(self):
        if not self.standing and (self.hop_only or self.walk_only):
            return [0.45, 0.55]

        self.last_clock_change = self.time
        if self.standing and np.random.randint(6) == 0:  # 1/6 chance of standing
            new_r = np.clip(np.random.normal(loc=0, scale=0.05), 0, min(self.get_info('ratio_bounds')))
            ratio = [new_r, 1 - new_r]

            self.currently_standing = True
        else:
            new_r      = np.random.uniform(*self.get_info('ratio_bounds'))
            ratio = [new_r, 1 - new_r]

            self.currently_standing = False
        return ratio

    def log_cmd(self):
        """
            Return a string with logging info for incoming command at this level of env. Used in interactive eval
        """

        out = \
        "Locomotion Planner:\n" + \
        "  ratio:\t{:3.2f},{:3.2f}\n".format(*self.cmd_dict['ratio']) + \
        "  shift:\t{:3.2f},{:3.2f}\n".format(*self.cmd_dict['period_shift']) + \
        "  freq: \t{:3.1f}\n".format(self.cmd_dict['step_freq']) + \
        "  turn_rate: \t{}\n".format(self.cmd_dict['turn_rate'])
        
        out += \
        "  x vel:\t{:5.2f}\n".format(self.cmd_dict['x_vel']) + \
        "  y vel:\t{:5.2f}\n".format(self.cmd_dict['y_vel'])

        return out
        
    def interactive_control(self, c):

        cmd = self.cmd_dict
        
        ##############
        # WASD group #
        ##############
        if c == 'w':
            cmd['x_vel'] += 0.1
        if c == 's':
            cmd['x_vel'] -= 0.1
        if c == 'd':
            cmd['y_vel'] += 0.1
        if c == 'a':
            cmd['y_vel'] -= 0.1
                
        if c == 'e':
            cmd['turn_rate'] -= 0.001 * np.pi/4
        if c == 'q':
            cmd['turn_rate'] += 0.001 * np.pi/4

        if c == 'o':
            cmd['step_freq'] = np.clip(self.cmd_dict['step_freq'] + 0.1, *self.get_info('step_freq_bounds'))
        if c == 'u':
            cmd['step_freq'] = np.clip(self.cmd_dict['step_freq'] - 0.1, *self.get_info('step_freq_bounds'))

        ###############
        # punct group #
        ###############
        if c == ']':
            cmd['ratio'][0] = np.clip(cmd['ratio'][0] + 0.01, 0.0, 1.0)
            cmd['ratio'][1] = np.clip(cmd['ratio'][1] - 0.01, 0.0, 1.0)
        if c == '[':
            cmd['ratio'][0] = np.clip(cmd['ratio'][0] - 0.01, 0.0, 1.0)
            cmd['ratio'][1] = np.clip(cmd['ratio'][1] + 0.01, 0.0, 1.0)
        if c == '.':
            cmd['period_shift'][0] = np.clip(cmd['period_shift'][0] - 0.05, -1.0, 1.0)
        if c == ';':
            cmd['period_shift'][0] = np.clip(cmd['period_shift'][0] + 0.05, -1.0, 1.0)
        if c == '/':
            cmd['period_shift'][1] = np.clip(cmd['period_shift'][1] - 0.05, -1.0, 1.0)
        if c == '\'':
            cmd['period_shift'][1] = np.clip(cmd['period_shift'][1] + 0.05, -1.0, 1.0)

        return cmd

    def interactive_control_gesture_cmd(self, c, forwardVel, backwardVel):

        cmd = self.cmd_dict
        
        ##############
        # WASD group #
        ##############

        if c == 'forward':
            cmd['x_vel'] = forwardVel
            cmd['y_vel'] = 0.0
            cmd['turn_rate'] = 0

        if c == 'backward':
            cmd['x_vel'] = -backwardVel
            cmd['y_vel'] = 0.0
            cmd['turn_rate'] = 0
        
        if c == 'rightStep':

            cmd['x_vel'] = 0.0
            cmd['y_vel'] = -backwardVel
            cmd['turn_rate'] = 0

        if c == 'leftStep':

            cmd['x_vel'] = 0.0
            cmd['y_vel'] = backwardVel
            cmd['turn_rate'] = 0

        if c == 'stop':
            cmd['y_vel'] = 0
            cmd['x_vel'] = 0
            cmd['turn_rate'] = 0
                
        if c == 'rightTurn':

            cmd['x_vel'] = backwardVel
            cmd['y_vel'] = 0.0
            cmd['turn_rate'] = -0.005 * np.pi/4
        if c == 'leftTurn':

            cmd['x_vel'] = backwardVel
            cmd['y_vel'] = 0.0
            cmd['turn_rate'] = 0.005 * np.pi/4

        if c == 'o':
            cmd['step_freq'] = np.clip(self.cmd_dict['step_freq'] + 0.1, *self.get_info('step_freq_bounds'))
        if c == 'u':
            cmd['step_freq'] = np.clip(self.cmd_dict['step_freq'] - 0.1, *self.get_info('step_freq_bounds'))

        ###############
        # punct group #
        ###############
        if c == ']':
            cmd['ratio'][0] = np.clip(cmd['ratio'][0] + 0.01, 0.0, 1.0)
            cmd['ratio'][1] = np.clip(cmd['ratio'][1] - 0.01, 0.0, 1.0)
        if c == '[':
            cmd['ratio'][0] = np.clip(cmd['ratio'][0] - 0.01, 0.0, 1.0)
            cmd['ratio'][1] = np.clip(cmd['ratio'][1] + 0.01, 0.0, 1.0)
        if c == '.':
            cmd['period_shift'][0] = np.clip(cmd['period_shift'][0] - 0.05, -1.0, 1.0)
        if c == ';':
            cmd['period_shift'][0] = np.clip(cmd['period_shift'][0] + 0.05, -1.0, 1.0)
        if c == '/':
            cmd['period_shift'][1] = np.clip(cmd['period_shift'][1] - 0.05, -1.0, 1.0)
        if c == '\'':
            cmd['period_shift'][1] = np.clip(cmd['period_shift'][1] + 0.05, -1.0, 1.0)

        return cmd


    def reset(self):
        """
            Reset for new episode.
        """
        self.reset_info()
        self.simulation_reset()
        self.cmd_dict = self.randomize_super_command(init=True)

        return self.get_full_state()

    def get_action_mirror_indices(self):
        mirror_act = [x for x in self.get_info('action_mirror_indices')]

        if self.auto_clock:
            mirror_act += [len(mirror_act)]

        return mirror_act

    def get_state_mirror_indices(self):
        mirror_obs = [x for x in self.get_info('state_mirror_indices')]
        mirror_obs += [len(mirror_obs) + i for i in range(2)] # ratio
        
        mirror_obs += [len(mirror_obs), -(len(mirror_obs)+1), -(len(mirror_obs)+2)] # xvel, yvel, turn rate

        if self.height_commanded:
            mirror_obs += len(mirror_obs)

        if self.perception:
            mirror_obs += [len(mirror_obs) + i for i in range(self.get_info('perception_size'))]

        if self.gaze_control:
            mirror_obs += [-len(mirror_obs), len(mirror_obs)+1]

        mirror_obs += [len(mirror_obs) + 1, len(mirror_obs)] # clock inputs
        
        return mirror_obs

    """ Functions past this point should be implemented in envs/cassie/cassie.py or
        envs/digit/digit.py, or another robot environment.
    """
    def get_robot_state(self):
        """
            Get full robot state. Used by locomotion planners.
        """
        raise NotImplementedError

    def randomize_dynamics(self):
        """
            Dynamics randomization at start of each new episode (for higher level RL envs)
        """
        raise NotImplementedError

    def default_dynamics(self):
        """
            Default Dynamics at start of each new episode (for higher level RL envs)
        """
        raise NotImplementedError

    def render(self):
        """
            Render mujoco simulator. This is to be called by robot-specific child classes.
        """
        # NOTE: Assumes self.vis is not None
        # print("Called")
        for i, (obj_id, obj) in enumerate(self.render_objs.items()):
            if obj['need_remove']:
                self.vis.remove_marker(i)
                del self.render_objs[obj_id]
                return
            if not obj['vized']:
                self.vis.add_marker(obj['pos'], obj['size'], obj['rgba'], obj['so3'])
                obj['vized'] = True
            if obj['need_update']:
                self.vis.update_marker(i, obj['pos'], obj['size'], obj['rgba'], obj['so3'])
                obj['need_update'] = False

    def add_render_obj(self, obj_id, pos, size, rgba, orient=None):
        if orient is None:
            orient = np.array([0., 0., 0.5])  # neutral forward orientation
        self.render_objs[obj_id] = {
            'pos': pos,
            'size' : size,
            'rgba' : rgba,
            'so3' : euler2so3(x=orient[0], y=orient[1], z=orient[2]).reshape(9),
            'vized' : False,
            'need_update' : False,
            'need_remove' : False
        }

    def remove_render_obj(self, obj_id):
        if obj_id in self.render_objs:
            self.render_objs[obj_id]['need_remove'] = True
        else:
            print(f"Warning: obj_id '{obj_id}' doesn't exist, can't remove!")

    def clear_render_objs(self):
        self.render_objs = {}
        self.vis.clear_markers()

    def update_render_obj(self, obj_id, pos=None, size=None, rgba=None, orient=None):
        if obj_id in self.render_objs:
            if pos is not None:
                self.render_objs[obj_id]['pos'] = pos
            if size is not None:
                self.render_objs[obj_id]['size'] = size
            if rgba is not None:
                self.render_objs[obj_id]['rgba'] = rgba
            if orient is not None:
                self.render_objs[obj_id]['so3'] = euler2so3(x=orient[0], y=orient[1], z=orient[2]).reshape(9)
            self.render_objs[obj_id]['need_update'] = True
        else:
            print(f"Warning: obj_id '{obj_id}' doesn't exist")
