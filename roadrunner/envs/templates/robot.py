import numpy as np
import random
from envs.terrains.stairs import Stairs

from util.quat import *

class RobotEnv:
    """This class abstracts out the information tracking, averaging, and collecting.
       Robot classes like CassieEnv or DigitEnv should inherit from this class.
    """
    def __init__(self, stairs=False,
                       simrate=50,
                       terrain=False,
                       **kwargs):

        self.stairs = stairs
        self.default_simrate = simrate
        self.terrain = terrain
        self.vis                       = None

        assert hasattr(self, 'robot_info_registry')
        assert hasattr(self, 'randomize_dynamics') or hasattr(self, 'default_dynamics')

        assert 'robot_state_size'     in self.robot_info_registry
        assert 'num_endeffectors'     in self.robot_info_registry
        assert 'state_mirror_indices' in self.robot_info_registry
        assert 'perception_size'      in self.robot_info_registry

        assert 'robot_foot_forces'     in self.robot_info_registry
        assert 'robot_foot_positions'  in self.robot_info_registry

        self.robot_info_registry['on_stairs'] = self.on_stairs

        self.trackers = [self.foot_force_tracker, self.foot_vel_tracker, self.motor_torque_tracker]

        self.stair_slats               = False #stairs and stair_slats
        self.stair_quat                = False #stairs and stair_quat
        if self.stairs:
            self.stairs = Stairs(self.sim, ['box1',
                                            'box2',
                                            'box3',
                                            'box4',
                                            'box5',
                                            'box6',
                                            'box7',
                                            'box8',
                                            'box9',
                                            'box10',
                                            'box11',
                                            'box12',
                                            'box13',
                                            'box14',
                                            'box15'])

    def get_info(self, attribute_name, *args, **kwargs):
        """
            Fetch info from info dict if it has been shared, if not raise an error.
        """
        if attribute_name in self.robot_info_registry:
            return self.robot_info_registry[attribute_name](*args, **kwargs)
        raise RuntimeError("RobotEnv.get_info() could not find '{}' in registry.".format(attribute_name))
   
    def set_info(self):
        raise NotImplementedError # TODO?

    def foot_force_tracker(self, n, final):
        """ This is an example of a tracker which averages data at high resolution.
            'n' is the number of times this tracker has been updated for this step().
            'final' is whether or not this will be the final update for this tracker.
        """
        if not hasattr(self, 'foot_frc_avg') or n == 0:
            self.foot_frc_avg = self.get_info('robot_foot_forces')
        else:
            self.foot_frc_avg = [0.984*x + 0.016*y for (x,y) in zip(self.foot_frc_avg, self.get_info('robot_foot_forces'))]
            self.robot_info_registry['smoothed_foot_forces'] = lambda: self.foot_frc_avg

    def motor_torque_tracker(self, n, final):
        if not hasattr(self, 'foot_frc_avg') or n == 0:
            self.torque_avg = self.get_info('robot_torque')
        else:
            self.torque_avg = [0.984*x + 0.016*y for (x,y) in zip(self.torque_avg, self.get_info('robot_torque'))]
            self.robot_info_registry['smoothed_torques'] = lambda: self.torque_avg

    def foot_vel_tracker(self, n, final):
        """ This is an example of a tracker which, despite being called at high resolution,
            will only update twice; in its first call, and in its final call.
        """
        if not hasattr(self, 'prev_foot_pos') or n == 0:
            self.prev_foot_pos = np.array(self.get_info('robot_foot_positions'))
        elif final:
            foot_pos = np.array(self.get_info('robot_foot_positions'))
            foot_vel = (foot_pos - self.prev_foot_pos) / ((n+1)/2000)
            self.prev_foot_pos = foot_pos

            vels = (np.sqrt(np.power(foot_vel[0], 2).sum()), np.sqrt(np.power(foot_vel[1], 2).sum()))
            self.robot_info_registry['smoothed_foot_velocities'] = lambda: vels

    def simulation_forward(self, action, n=50):
        """ This function runs the low-level simulation forward n steps,
            while running update_tracker() functions set by inheriting classes.
        """
        for i in range(n):
            """ step_simulation is implemented in the inheriting CassieEnv or
                DigitEnv, or some other robot environment.
            """
            self.step_simulation(action)

            for fn in self.trackers:
                """ Update each tracker function in self.trackers
                """
                fn(i, i == n-1)

    def simulation_forward_with_logging(self, action, n=50):
        """ This function runs the low-level simulation forward n steps,
            while running update_tracker() functions set by inheriting classes.
        """
        cassie_logData = np.empty([n,77],dtype=object)        
        for i in range(n):
            """ step_simulation is implemented in the inheriting CassieEnv or
                DigitEnv, or some other robot environment.
            """
            self.step_simulation(action)
            
            translational_vel = self.get_robot_translational_velocity(estimate = False)[0]
            left_foot_force = np.array(self.sim.get_foot_forces())[0]
            right_foot_force = np.array(self.sim.get_foot_forces())[1]
            cassie_pose = tuple((self.sim.qpos(),self.sim.qvel(), left_foot_force, right_foot_force, translational_vel, self.phase))
            cassie_logData[i,0] = "qpos"
            cassie_logData[i,1:36] = cassie_pose[0]
            cassie_logData[i,36] = "qvel"
            cassie_logData[i,37:69] = cassie_pose[1]
            cassie_logData[i,69] = "left foot force"
            cassie_logData[i,70] = cassie_pose[2]
            cassie_logData[i,71] = "right foot force"
            cassie_logData[i,72] = cassie_pose[3]
            cassie_logData[i,73] = "pelvis vel"
            cassie_logData[i,74] = cassie_pose[4]
            cassie_logData[i,75] = "phase angle"
            cassie_logData[i,76] = cassie_pose[5]
            
            for fn in self.trackers:
                """ Update each tracker function in self.trackers
                """
                fn(i, i == n-1)

        return cassie_logData 
    def get_ground_height(self, x, y, set_marker=False):
        x_scale, y_scale, max_z, base_z = self.sim.get_hfield_size()
        z_scale = max_z - base_z

        if x > x_scale or x < -x_scale or y > y_scale or y < -y_scale:
            return np.inf

        x_res, y_res = self.sim.get_hfield_nrow(), self.sim.get_hfield_ncol()

        data = self.sim.get_hfield_data().reshape((x_res, y_res))

        if True:
          x_pixel = x_res // 2 + int(x / x_scale * x_res / 2)
          y_pixel = y_res // 2 + int(y / y_scale * y_res / 2)
        else:
          x_pixel = int(x / x_scale * x_res)
          y_pixel = int(y / y_scale * y_res)

        return base_z + z_scale * data[y_pixel, x_pixel]

    def on_stairs(self):
        if self.stairs:
            qpos = self.sim.qpos()
            pelvis_step, _ = self.stairs.check_step(*qpos[:3])
            if qpos[0] > self.stairs.bounds[0] and \
               qpos[0] < self.stairs.bounds[1] and \
               pelvis_step not in self.stairs.exempt_stairs:
                return True
        else:
            return False


    def rotate_to_heading(self, vec, **kwargs):
        quaternion  = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def simulation_reset(self, **kwargs):
        # Randomize dynamics:
        if self.dynamics_randomization:
            self.randomize_dynamics()
        else:
            self.default_dynamics()

        qpos = self.sim.qpos()
        spawn_point = np.array([np.random.uniform(-1, 1), np.random.uniform(-3, 3), qpos[2]])
        if self.terrain:
            self.sim.randomize_terrain()
            """
                Check spawn location to make sure feet aren't clipped into terrain.
            """
            left_foot_pos  = np.array(spawn_point + self.sim.foot_pos()[:3])
            right_foot_pos = np.array(spawn_point + self.sim.foot_pos()[3:])
            z_deltas = []
            for (x,y,z) in [left_foot_pos, right_foot_pos]:
                z -= qpos[2]
                heel_hgt = self.get_ground_height(x - 0.06, y)
                toe_hgt = self.get_ground_height(x + 0.1, y)
                z_hgt = max(heel_hgt, toe_hgt)
                z_deltas.append((z_hgt-z))

            spawn_point[2] += max(z_deltas)

        if self.stairs:
            """
                This creates the stairs.
            """
            start_x   = np.random.uniform(-1, 2)
            stair_len = np.random.uniform(0.24, 0.37)
            stair_hgt = np.random.uniform(0.1, 0.17)
            height    = np.random.choice(a=[1, 2, 3, 4, 5, 6, 7, 8])

            if self.stair_slats:
                slats = np.random.randint(3) == 0
            else:
                slats = False

            if self.stair_quat:
                quat = np.random.randint(2) == 0
            else:
                quat = False

            ground_hgt = 0
            if self.terrain:
                ground_hgt = self.get_ground_height(spawn_point[0], spawn_point[1])

            self.stairs.create_stairs(spawn_point[0] + start_x, spawn_point[1], 0, stair_len, stair_hgt, height=height, slats=slats, quat=quat)

            """
                This makes sure the robot does not clip into the stairs.
            """
            left_foot_pos  = np.array(spawn_point + self.sim.foot_pos()[:3]) - np.array((0, 0, qpos[2]))
            right_foot_pos = np.array(spawn_point + self.sim.foot_pos()[3:]) - np.array((0, 0, qpos[2]))
            z_deltas = []
            for (x,y,z) in [left_foot_pos, right_foot_pos]:
                _, heel_hgt = self.stairs.check_step(x - 0.06, y, 0)
                _, toe_hgt  = self.stairs.check_step(x + 0.1, y, 0)
                z_hgt    = max(heel_hgt, toe_hgt)
                z_deltas.append((z_hgt-z))

            delta = max(z_deltas)
            if delta > spawn_point[2] - qpos[2]:
                spawn_point[2] += delta
        else:
            pass
            # Reset to starting position via IK
            #if hasattr(self, 'IK_solver'):
            #    self.sim.set_qpos(self.IK_solver(*get_sample_pos()))

        # Reset to starting position via IK
        #if hasattr(self, 'IK_solver'):
        #    self.sim.set_qpos(self.IK_solver(*get_sample_pos()))

        if self.vis is not None: # needed to update visualization hfield
            self.vis.reset(self.sim)

        self.sim.set_qpos(np.hstack((spawn_point, qpos[3:])))
        self.robot_state = self.sim.step_pd(self.u)


