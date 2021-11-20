"""
    Implements the ICRA 2021 periodic reward calculation.
    Assumes that 'self' extends PeriodicEnv.
"""
import numpy as np
from util.quat import *
from util.clock import Phase, vonmises_func

def kernel(x):
  return np.exp(-np.abs(x)) - 1

def setup_reward_components(self, incentive=False):
    if incentive:
        coeff       = [-1, 1]
    else:
        coeff       = [-1, 0]

    w = {}

    self.add_reward_component('left force',  0, coeff=coeff)
    self.add_reward_component('right force', 1, coeff=coeff)
    self.add_reward_component('left spd',    0, coeff=coeff[::-1])
    self.add_reward_component('right spd',   1, coeff=coeff[::-1])
    self.add_reward_component('x vel')
    self.add_reward_component('y vel')
    self.add_reward_component('orientation')

    w['left force']  = 0.125
    w['right force'] = 0.125
    w['left spd']    = 0.125
    w['right spd']   = 0.125
    w['x vel'] = 0.125
    w['y vel'] = 0.075
    w['orientation']  = 0.125

    if self.antistomp:
        self.add_reward_component('left antistomp')
        self.add_reward_component('right antistomp')

        w['left antistomp']  = 0.125
        w['right antistomp'] = 0.125


    if not self.walk_only and not self.hop_only:
        self.add_reward_component('hop symmetry')
        w['hop symmetry'] = 0.100

    if self.standing:
        self.add_reward_component('standing cost')
        w['standing cost'] = 0.1

    if self.height_commanded:
        self.add_reward_component('height cost')
        w['height cost'] = 0.1

    if self.stairs and self.use_stair_heuristic:
        self.add_reward_component('left stair cost',  0, coeff=coeff[::-1])
        self.add_reward_component('right stair cost',  1, coeff=coeff[::-1])
        w['left stair cost'] = 0.1
        w['right stair cost'] = 0.1

    self.add_reward_component('pelvis acc')
    self.add_reward_component('ctrl')
    self.add_reward_component('torque')

    w['pelvis acc'] = 0.025
    w['ctrl']       = 0.025
    w['torque']     = 0.025

    total = sum(w.values())

    for name in w:
        w[name] = w[name] / total
    
    return w

def compute_done(self):
    done = False

def compute_reward(self, action):
    grounded_sum = sum(self.behavior.ratio[i] for i in range(0, self.num_phases, 2))
    if self.standing:
        omega = 1 / (1 + np.exp(-50 * (grounded_sum - 0.15)))
    else:
        omega = 1
    pdif = np.exp(-5 * np.abs(np.sin(np.pi * (self.behavior.period_shift[0] - self.behavior.period_shift[1]))))
    robot_pos = self.get_info('robot_position')

    q = {}

    lfrc, rfrc = self.get_info('smoothed_foot_forces')
    lvel, rvel = self.get_info('smoothed_foot_velocities')
    q['left force']  = kernel(omega * np.abs(lfrc) / (100))
    q['right force'] = kernel(omega * np.abs(rfrc) / (100))

    q['left spd']  = kernel(omega * np.abs(lvel))
    q['right spd'] = kernel(omega * np.abs(rvel))

    if self.antistomp:
        q['left antistomp'] = kernel(self.peak_grf_change[0]/100)
        q['right antistomp'] = kernel(self.peak_grf_change[1]/100)
        #print(q['left antistomp'], q['right antistomp'])

    if self.height_commanded:
        pelvis_hgt = np.abs(self.get_info('robot_height') - self.height) * 3

        if pelvis_hgt < 0.02:
            pelvis_hgt = 0
        q['height cost'] = kernel(pelvis_hgt)

    pelvis_vel = self.get_info('robot_translational_velocity', rotate_to_heading=True)
    x_vel = np.abs(pelvis_vel[0] - self.cmd_dict['x_vel'])
    if x_vel < 0.02:
        x_vel = 0

    y_vel = np.abs(pelvis_vel[1] - self.cmd_dict['y_vel'])
    if y_vel < 0.02:
        y_vel = 0

    q['x vel'] = kernel(x_vel * 2 * omega)
    q['y vel'] = kernel(y_vel * 2 * omega)

    actual_q = self.get_info('robot_orientation', rotate_to_heading=True)

    if self.gaze_control:
        target_q = euler2quat(z=self.gaze_yaw, y=0, x=self.gaze_pitch)
        left_actual, right_actual  = self.get_info('robot_foot_orientations', local=True, rotate_by_quat=target_q)
    else:
        target_q = [1, 0, 0, 0]
        left_actual, right_actual  = self.get_info('robot_foot_orientations', local=False)
    orientation_error = 3 * (1 - np.inner(actual_q, target_q) ** 2)

    liftoff_target = 0.1
    touchdown_target = -0.25 * np.pi
    #if np.abs(self.phase - self.liftoff_point[0]) < self.phase_add:
    #  print("near left liftoff!")
    #if np.abs(self.phase - self.touchdown_point[0]) < self.phase_add:
    #  print("near left touchdown!")
    #if np.abs(self.phase - self.liftoff_point[1]) < self.phase_add:
    #  print("near right liftoff!")
    #if np.abs(self.phase - self.touchdown_point[1]) < self.phase_add:
    #  print("near right touchdown!")

    ltw = np.exp(-4 * np.abs(np.sin(np.pi * (self.phase - self.touchdown_point[0])/self.phase_len)))
    llw = np.exp(-4 * np.abs(np.sin(np.pi * (self.phase - self.liftoff_point[0])/self.phase_len)))
    ldw = 1 - (ltw + llw)
    ltw, llw, ldw = np.array((ltw, llw, ldw)) / sum([ltw, llw, ldw])
    left_pitch = quaternion2euler(left_actual)[1]

    rtw = np.exp(-4 * np.abs(np.sin(np.pi * (self.phase - self.touchdown_point[1])/self.phase_len)))
    rlw = np.exp(-4 * np.abs(np.sin(np.pi * (self.phase - self.liftoff_point[1])/self.phase_len)))
    rdw = 1 - (rtw + rlw)
    rtw, rlw, rdw = np.array((rtw, rlw, rdw)) / sum([rtw, rlw, rdw])
    right_pitch = quaternion2euler(right_actual)[1]

    left_pitch_target  = ltw * touchdown_target + llw * liftoff_target * ldw * left_pitch
    right_pitch_target = rtw * touchdown_target + rlw * liftoff_target * rdw * right_pitch

    left_actual_target_euler  = [0, left_pitch_target, 0]
    right_actual_target_euler = [0, right_pitch_target, 0]

    #print('left: touchdown {:3.2f}, liftoff {:3.2f}, default {:3.2f}, target {:4.3f}, actual {:4.3f}'.format(ltw, llw, ldw, left_pitch_target, left_pitch))
    #print('right: touchdown {:3.2f}, liftoff {:3.2f}, default {:3.2f}, target {:4.3f}, actual {:4.3f}'.format(rtw, rlw, rdw, right_pitch_target, right_pitch))
    #input()

    left_actual_target  = euler2quat(z=left_actual_target_euler[2], y=left_actual_target_euler[1], x=left_actual_target_euler[0])
    right_actual_target = euler2quat(z=right_actual_target_euler[2], y=right_actual_target_euler[1], x=right_actual_target_euler[0])

    foot_err = 10 * ((1 - np.inner(left_actual, left_actual_target) ** 2) + (1 - np.inner(right_actual, right_actual_target) ** 2))
    q['orientation'] = kernel(orientation_error + foot_err)

    rvel = self.get_info('robot_angular_velocity')
    tacc = self.get_info('robot_translational_acceleration')
    q['pelvis acc'] = kernel(0.10 * (np.abs(rvel).sum() + np.abs(tacc).sum()))

    # Torque cost term
    q['ctrl']   = kernel(self.avg_ctrl_penalty)
    q['torque'] = kernel(self.avg_trq_penalty)

    lpos, rpos = self.get_info('robot_foot_positions', local=True)
    lpos = np.array([lpos[0], lpos[2]])
    rpos = np.array([rpos[0], rpos[2]])
    xdif = 10 * np.sqrt(np.power(lpos - rpos, 2).sum())

    if not self.walk_only and not self.hop_only:
        q['hop symmetry'] = kernel(pdif * xdif)

    if self.standing:
        q['standing cost'] = kernel((xdif + 20 * self.avg_ctrl_penalty) * (1 - omega))

    l_foot_pos, r_foot_pos = self.get_info('robot_foot_positions', local=False)
    if self.stairs: # make sure the policy takes exactly one step per stair.
        pelvis_step, _ = self.stairs.check_step(*robot_pos[:3])
        if robot_pos[0] > self.stairs.bounds[0] and \
           robot_pos[0] < self.stairs.bounds[1] and \
           pelvis_step not in self.stairs.exempt_stairs:
              q['x vel'] *= 0.5
              q['y vel'] *= 0.5

        if self.use_stair_heuristic:
            q['left stair cost'] = 0
            q['right stair cost'] = 0
            if self.time > 15:
              last_left, last_right = self.last_stair_step
              next_left, next_right = self.next_stair_step

              l_stance, r_stance = self.foot_in_stance

              for foot_pos, last_step, next_step, in_stance, f in [(l_foot_pos, last_left, next_left, l_stance, 'left stair cost'),
                                                                   (r_foot_pos, last_right, next_right, r_stance, 'right stair cost')]:
                  foot_step, caught_edge = self.stairs.check_foot_step(*foot_pos)
                  if foot_step is not None and next_step is not None and next_step < len(self.stairs.available_geoms):
                      if not (foot_step in self.stairs.exempt_stairs and next_step in self.stairs.exempt_stairs):
                          if foot_step != next_step:
                              box_x, _, _= self.sim.get_geom_pos(self.stairs.available_geoms[next_step])
                              factor = 4 if caught_edge else 2
                              q[f] = kernel(factor*np.abs(foot_pos[0] - box_x))/2

    r = self.periodic_reward(q, self.w, b=1, debug=False)
    return r

def compute_done(self):
    actual_q = self.get_info('robot_orientation', rotate_to_heading=True)

    if self.gaze_control:
        target_q = euler2quat(z=self.gaze_yaw, y=0, x=self.gaze_pitch)
        left_actual, right_actual  = self.get_info('robot_foot_orientations', local=True, rotate_by_quat=target_q)
    else:
        target_q = [1, 0, 0, 0]
        left_actual, right_actual  = self.get_info('robot_foot_orientations', local=False)

    orientation_error = 3 * (1 - np.inner(actual_q, target_q) ** 2)
    if np.exp(-orientation_error) < 0.8:
        return True
    else:
        return False

