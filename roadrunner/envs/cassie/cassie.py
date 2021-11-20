import numpy as np
import time

from envs.cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from envs.templates.robot import RobotEnv

from util.quat import *

class CassieEnv(RobotEnv):
    def __init__(self, *args, **kwargs):
        self.perception_size = 6
        self.state_est_size  = 35
        self.num_actuators   = 10
        self.num_endeffectors = 2  # Info for HRL learning taskspace
        self.taskspace_dim   = 12  # Info for HRL learning taskspace

        self.sim = CassieSim(terrain=kwargs['terrain'], perception=kwargs['perception'])

        self.pos_idx      = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.P            = np.array([100,  100,  88,  96,  50])
        self.D            = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.offset       = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.task_offset  = np.array([0, 0.135, -0.9, 0, -0.135, -0.9])
        self.u            = pd_in_t()
        self.robot_state = state_out_t()

        self.simrate_bounds = [0.95, 1.05]
        self.incline_bounds = [-0.03, 0.03]
        self.encoder_noise  = 0.05
        self.damping_bounds = [0.5, 3.5]
        self.mass_bounds    = [0.5, 1.7]
        self.fric_bounds    = [0.35, 1.1]

        self.speed_bounds      = [-0.2, 2.0]
        self.side_speed_bounds = [-0.3, 0.3]
        self.step_freq_bounds  = [0.9, 1.3]
        self.gaze_bounds       = [-0.2, 0.2]
        self.height_bounds     = [0.7, 1.0]
        self.ratio_bounds      = [0.35, 0.70]

        # Record default dynamics parameters
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass    = self.sim.get_body_mass()
        self.default_ipos    = self.sim.get_body_ipos()
        self.default_fric    = self.sim.get_geom_friction()
        self.default_rgba    = self.sim.get_geom_rgba()
        self.default_quat    = self.sim.get_geom_quat()

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

        self.robot_state_mirror_indices = [0.01, -1, 2, -3,      # pelvis orientation
                                          -4, 5, -6,             # rotational vel
                                          -12, -13, 14, 15, 16,  # left motor pos
                                          -7,  -8,  9,  10,  11, # right motor pos
                                          -22, -23, 24, 25, 26,  # left motor vel
                                          -17, -18, 19, 20, 21,  # right motor vel 
                                          29, 30, 27, 28,        # joint pos
                                          33, 34, 31, 32, ]      # joint vel
        
        self.robot_action_mirror_indices = [-5, -6, 7, 8, 9,
                                            -0.1, -1, 2, 3, 4]

        if kwargs['impedance']:
            self.robot_action_mirror_indices += [15, 16, 17, 18, 19,
                                                 10, 11, 12, 13, 14,
                                                 25, 26, 27, 28, 29,
                                                 20, 21, 22, 23, 24]

        # self.robot_const_registry['taskspace_dim'] = self.taskspace_dim
        self.robot_info_registry = {}
        self.robot_info_registry['robot_state_size']      = lambda: self.state_est_size
        self.robot_info_registry['num_endeffectors']      = lambda: self.num_endeffectors
        self.robot_info_registry['task_offset']           = lambda: self.task_offset
        self.robot_info_registry['state_mirror_indices']  = lambda: self.robot_state_mirror_indices
        self.robot_info_registry['action_mirror_indices'] = lambda: self.robot_action_mirror_indices
        self.robot_info_registry['perception_size']       = lambda: self.perception_size
        
        self.robot_info_registry['simrate_bounds']   = lambda: self.simrate_bounds
        self.robot_info_registry['incline_bounds']   = lambda: self.incline_bounds
        self.robot_info_registry['encoder_noise']    = lambda: self.encoder_noise
        self.robot_info_registry['damping_bounds']   = lambda: self.damping_bounds
        self.robot_info_registry['mass_bounds']      = lambda: self.mass_bounds

        self.robot_info_registry['speed_bounds']     = lambda: self.speed_bounds
        self.robot_info_registry['side_speed_bounds']= lambda: self.side_speed_bounds
        self.robot_info_registry['step_freq_bounds'] = lambda: self.step_freq_bounds
        self.robot_info_registry['gaze_bounds']      = lambda: self.gaze_bounds
        self.robot_info_registry['height_bounds']    = lambda: self.height_bounds
        self.robot_info_registry['ratio_bounds']     = lambda: self.ratio_bounds

        self.robot_info_registry['robot_state']                      = self.get_robot_state
        self.robot_info_registry['robot_orientation']                = self.get_robot_orientation
        self.robot_info_registry['robot_position']                   = self.get_robot_position
        self.robot_info_registry['robot_translational_velocity']     = self.get_robot_translational_velocity
        self.robot_info_registry['robot_translational_acceleration'] = self.get_robot_translational_acceleration
        self.robot_info_registry['robot_angular_velocity']           = self.get_robot_angular_velocity
        self.robot_info_registry['robot_foot_positions']             = self.get_robot_foot_positions
        self.robot_info_registry['robot_foot_forces']                = self.get_robot_foot_forces
        self.robot_info_registry['robot_foot_orientations']          = self.get_robot_foot_orientations
        self.robot_info_registry['robot_motor_positions']            = self.get_robot_motor_positions
        self.robot_info_registry['robot_height']                     = self.get_robot_height

        self.robot_info_registry['robot_perception'] = self.get_perception
        self.robot_info_registry['robot_torque'] = self.get_torques

        RobotEnv.__init__(self, **kwargs)

    def step_simulation(self, action):
        target = action[:10] + self.offset
        p_add  = np.zeros(10)
        d_add  = np.zeros(10)

        if len(action) > 10:
            p_add = action[10:20]

        if len(action) > 20:
            d_add = action[20:30]

        if self.dynamics_randomization:
            target -= self.motor_encoder_noise

        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i] + p_add[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i] + p_add[i + 5]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i] + d_add[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i] + d_add[i + 5]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.robot_state = self.sim.step_pd(self.u)

    def randomize_dynamics(self):
        damp = self.default_damping
        pelvis_damp_range = [[damp[0], damp[0]],
                            [damp[1], damp[1]],
                            [damp[2], damp[2]],
                            [damp[3], damp[3]],
                            [damp[4], damp[4]],
                            [damp[5], damp[5]]]  # 0->5

        hip_damp_range = [[damp[6]*self.damping_bounds[0], damp[6]*self.damping_bounds[1]],
                          [damp[7]*self.damping_bounds[0], damp[7]*self.damping_bounds[1]],
                          [damp[8]*self.damping_bounds[0], damp[8]*self.damping_bounds[1]]]        # 6->8 and 19->21

        achilles_damp_range = [[damp[9]*self.damping_bounds[0],  damp[9]*self.damping_bounds[1]],
                               [damp[10]*self.damping_bounds[0], damp[10]*self.damping_bounds[1]],
                               [damp[11]*self.damping_bounds[0], damp[11]*self.damping_bounds[1]]]  # 9->11 and 22->24

        knee_damp_range   = [[damp[12]*self.damping_bounds[0], damp[12]*self.damping_bounds[1]]]  # 12 and 25
        shin_damp_range   = [[damp[13]*self.damping_bounds[0], damp[13]*self.damping_bounds[1]]]  # 13 and 26
        tarsus_damp_range = [[damp[14]*self.damping_bounds[0], damp[14]*self.damping_bounds[1]]]  # 14 and 27

        heel_damp_range   = [[damp[15], damp[15]]]                                     # 15 and 28
        fcrank_damp_range = [[damp[16]*self.damping_bounds[0], damp[16]*self.damping_bounds[1]]]  # 16 and 29
        prod_damp_range   = [[damp[17], damp[17]]]                                     # 17 and 30
        foot_damp_range   = [[damp[18]*self.damping_bounds[0], damp[18]*self.damping_bounds[1]]]  # 18 and 31

        side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
        damp_range = pelvis_damp_range + side_damp + side_damp
        damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

        m = self.default_mass
        pelvis_mass_range      = [[self.mass_bounds[0]*m[1],  self.mass_bounds[1]*m[1]]]  # 1
        hip_mass_range         = [[self.mass_bounds[0]*m[2],  self.mass_bounds[1]*m[2]],  # 2->4 and 14->16
                                    [self.mass_bounds[0]*m[3],  self.mass_bounds[1]*m[3]],
                                    [self.mass_bounds[0]*m[4],  self.mass_bounds[1]*m[4]]]

        achilles_mass_range    = [[self.mass_bounds[0]*m[5],  self.mass_bounds[1]*m[5]]]  # 5 and 17
        knee_mass_range        = [[self.mass_bounds[0]*m[6],  self.mass_bounds[1]*m[6]]]  # 6 and 18
        knee_spring_mass_range = [[self.mass_bounds[0]*m[7],  self.mass_bounds[1]*m[7]]]  # 7 and 19
        shin_mass_range        = [[self.mass_bounds[0]*m[8],  self.mass_bounds[1]*m[8]]]  # 8 and 20
        tarsus_mass_range      = [[self.mass_bounds[0]*m[9],  self.mass_bounds[1]*m[9]]]  # 9 and 21
        heel_spring_mass_range = [[self.mass_bounds[0]*m[10], self.mass_bounds[1]*m[10]]] # 10 and 22
        fcrank_mass_range      = [[self.mass_bounds[0]*m[11], self.mass_bounds[1]*m[11]]] # 11 and 23
        prod_mass_range        = [[self.mass_bounds[0]*m[12], self.mass_bounds[1]*m[12]]] # 12 and 24
        foot_mass_range        = [[self.mass_bounds[0]*m[13], self.mass_bounds[1]*m[13]]] # 13 and 25

        side_mass = hip_mass_range + achilles_mass_range \
                    + knee_mass_range + knee_spring_mass_range \
                    + shin_mass_range + tarsus_mass_range \
                    + heel_spring_mass_range + fcrank_mass_range \
                    + prod_mass_range + foot_mass_range

        mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
        mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

        delta = 0.0
        com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

        fric_noise = []
        translational = np.random.uniform(*self.fric_bounds)
        torsional = np.random.uniform(1e-4, 5e-4)
        rolling = np.random.uniform(1e-4, 2e-4)
        for _ in range(int(len(self.default_fric)/3)):
            fric_noise += [translational, torsional, rolling]

        geom_plane = [np.random.uniform(*self.incline_bounds), np.random.uniform(*self.incline_bounds), 0]
        quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
        geom_quat  = list(quat_plane) + list(self.default_quat[4:])

        self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
        self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)

        self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
        self.sim.set_body_mass(np.clip(mass_noise, 0, None))
        self.sim.set_body_ipos(com_noise)
        self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
        if self.stairs:
            self.sim.set_geom_quat(self.default_quat)
        else:
            self.sim.set_geom_quat(geom_quat)
        self.pd_simrate = int(self.default_simrate)# * np.random.uniform(self.min_simrate, self.max_simrate))

        self.sim.set_const()

    def default_dynamics(self):
        self.sim.set_body_mass(self.default_mass)
        self.sim.set_body_ipos(self.default_ipos)
        self.sim.set_dof_damping(self.default_damping)
        self.sim.set_geom_friction(self.default_fric)
        self.sim.set_geom_quat(self.default_quat)
        self.pd_simrate = int(self.default_simrate)

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

        self.sim.set_const()

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim)
            #time.sleep(0.01)
        #super().render()
        #time.sleep(0.01)
        return self.vis.draw(self.sim)

    def get_robot_state(self):
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
            self.robot_state.pelvis.rotationalVelocity[:],              # pelvis rotational velocity 
            motor_pos,                                                   # actuated joint positions
            motor_vel,                                                   # actuated joint velocities
            joint_pos,                                                   # unactuated joint positions
            joint_vel                                                    # unactuated joint velocities
        ])

        return robot_state

    def get_robot_foot_positions(self, **kwargs):
        return np.array(self.sim.foot_pos())

    def get_perception(self, **kwargs):
        return np.array(self.sim.sense_ground())

    def get_torques(self, **kwargs):
        return np.asarray(self.robot_state.motor.torque[:])

    def get_robot_height(self, naive=False, **kwargs):
        if naive: # this is a bad idea, do not do this
            return self.sim.qpos()[2]

        #return self.sim.qpos()[2] - self.get_ground_height(*self.sim.qpos()[:2])
        raise NotImplementedError

    def get_robot_orientation(self, rotate_to_heading=True, **kwargs):
        quat = self.robot_state.pelvis.orientation[:]
        if rotate_to_heading:
            return self.rotate_to_heading(quat)
        else:
            return quat

    def get_robot_translational_velocity(self, rotate_to_heading=True, estimate=True, **kwargs):
        if estimate:
          vel = self.robot_state.pelvis.translationalVelocity[:]
        else:
          vel = np.array(self.sim.qvel()[:3])

        if rotate_to_heading:
            return self.rotate_to_heading(vel)
        else:
            return vel

    def get_robot_translational_acceleration(self, **kwargs):
        return self.robot_state.pelvis.translationalAcceleration[:]

    def get_robot_angular_velocity(self, **kwargs):
        return self.robot_state.pelvis.rotationalVelocity[:]

    def get_robot_foot_positions(self, local=False, **kwargs):
        if local:
            return self.robot_state.leftFoot.position[:], self.robot_state.rightFoot.position[:]
        return np.array(self.sim.foot_pos())[0:3], np.array(self.sim.foot_pos())[3:6]
    
    def get_robot_foot_forces(self, **kwargs):
        return self.sim.get_foot_forces()

    def get_robot_foot_orientations(self, local=True, rotate_by_quat=None, **kwargs):
        lquat = self.robot_state.leftFoot.orientation[:]
        rquat = self.robot_state.rightFoot.orientation[:]
        if not local:
            ref = self.get_robot_orientation()
            lquat = quaternion_product(ref, lquat)
            rquat = quaternion_product(ref, rquat)

        if rotate_by_quat is not None:
            lquat = quaternion_product(rotate_by_quat, lquat)
            rquat = quaternion_product(rotate_by_quat, rquat)

        return lquat, rquat

    def get_robot_position(self, **kwargs):
        return np.array(self.sim.qpos()[:3])

    def get_robot_velocity(self, estimate=True, **kwargs):
        if not estimate:
            return np.array(self.sim.qvel()[:3])
        return self.robot_state.pelvis.translationalVelocity[:]

    def get_robot_motor_positions(self, estimate=True, **kwargs):
        if estimate:
            return self.robot_state.motor.position[:]
        return self.sim.qpos()[self.pos_idx]

    def get_robot_heading(self):
        return self.orient_add
            
    def log_cmd(self):
        """
            Return a string with logging info for incoming command at this level of env. Used in interactive eval
        """
        pos = self.get_robot_position()
        vel = self.get_robot_velocity(estimate=False)
        out = \
        "Cassie:\n" + \
        "  pos:   \t{:3.2f},{:3.2f}\n".format(pos[0],pos[1]) + \
        "  orient:\t{:5.2f}\n".format(self.get_robot_heading()) + \
        "  x vel:\t{:5.2f}\n".format(vel[0]) + \
        "  y vel:\t{:5.2f}\n".format(vel[1])

        return super().log_cmd() + out

#nbody layout:
# 0:  worldbody (zero)
# 1:  pelvis

# 2:  left hip roll 
# 3:  left hip yaw
# 4:  left hip pitch
# 5:  left achilles rod
# 6:  left knee
# 7:  left knee spring
# 8:  left shin
# 9:  left tarsus
# 10:  left heel spring
# 12:  left foot crank
# 12: left plantar rod
# 13: left foot

# 14: right hip roll 
# 15: right hip yaw
# 16: right hip pitch
# 17: right achilles rod
# 18: right knee
# 19: right knee spring
# 20: right shin
# 21: right tarsus
# 22: right heel spring
# 23: right foot crank
# 24: right plantar rod
# 25: right foot


# qpos layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation qw
# [ 4] Pelvis orientation qx
# [ 5] Pelvis orientation qy
# [ 6] Pelvis orientation qz
# [ 7] Left hip roll         (Motor [0])
# [ 8] Left hip yaw          (Motor [1])
# [ 9] Left hip pitch        (Motor [2])
# [10] Left achilles rod qw
# [11] Left achilles rod qx
# [12] Left achilles rod qy
# [13] Left achilles rod qz
# [14] Left knee             (Motor [3])
# [15] Left shin                        (Joint [0])
# [16] Left tarsus                      (Joint [1])
# [17] Left heel spring
# [18] Left foot crank
# [19] Left plantar rod
# [20] Left foot             (Motor [4], Joint [2])
# [21] Right hip roll        (Motor [5])
# [22] Right hip yaw         (Motor [6])
# [23] Right hip pitch       (Motor [7])
# [24] Right achilles rod qw
# [25] Right achilles rod qx
# [26] Right achilles rod qy
# [27] Right achilles rod qz
# [28] Right knee            (Motor [8])
# [29] Right shin                       (Joint [3])
# [30] Right tarsus                     (Joint [4])
# [31] Right heel spring
# [32] Right foot crank
# [33] Right plantar rod
# [34] Right foot            (Motor [9], Joint [5])

# qvel layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation wx
# [ 4] Pelvis orientation wy
# [ 5] Pelvis orientation wz
# [ 6] Left hip roll         (Motor [0])
# [ 7] Left hip yaw          (Motor [1])
# [ 8] Left hip pitch        (Motor [2])
# [ 9] Left knee             (Motor [3])
# [10] Left shin                        (Joint [0])
# [11] Left tarsus                      (Joint [1])
# [12] Left foot             (Motor [4], Joint [2])
# [13] Right hip roll        (Motor [5])
# [14] Right hip yaw         (Motor [6])
# [15] Right hip pitch       (Motor [7])
# [16] Right knee            (Motor [8])
# [17] Right shin                       (Joint [3])
