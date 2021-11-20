try:
  from digitmujoco_ctypes import *
except ImportError:
  from .digitmujoco_ctypes import *

import os
import ctypes
import numpy as np

from envs.terrains.rand import *

# Get base directory
_dir_path = os.path.dirname(os.path.realpath(__file__))

# Initialize libdigitsim
digit_mujoco_init(str.encode(_dir_path+"/digit.xml"))

class DigitSim:
    def __init__(self, terrain=False, perception=False, reinit=False):

        base = 'digit'
        if perception:
            base += '_perception'
        if terrain:
            base += '_hfield'

        modelfile = os.path.join(_dir_path, base + '.xml')

        self.c = digit_sim_init(modelfile.encode('utf-8'), True)

        if terrain:
            x_res, y_res = self.get_hfield_nrow(), self.get_hfield_ncol()
            self.hfields = generate_perlin(x_res, y_res)
            
        self.nv    = digit_sim_nv(self.c)
        self.nbody = digit_sim_nbody(self.c)
        self.nq    = digit_sim_nq(self.c)
        self.ngeom = digit_sim_ngeom(self.c)

    def randomize_terrain(self):
        hfield = self.hfields[np.random.randint(len(self.hfields))]
        self.set_hfield_data(hfield.flatten())

    def step_pd(self, u):
        digit_sim_step_pd(self.c, u)
        state = digit_sim_out(self.c)
        return state.contents

    def time(self):
        timep = digit_sim_time(self.c)
        return timep[0]

    def qpos(self):
        qposp = digit_sim_qpos(self.c)
        return qposp[:self.nq]

    def qvel(self):
        qvelp = digit_sim_qvel(self.c)
        return qvelp[:self.nv]

    def qacc(self):
        qaccp = digit_sim_qacc(self.c)
        return qaccp[:self.nv]

    def xquat(self, body_name):
        xquatp = digit_sim_xquat(self.c, body_name.encode())
        return xquatp[:4]

    def set_time(self, time):
        timep = digit_sim_time(self.c)
        timep[0] = time

    def set_qpos(self, qpos):
        qposp = digit_sim_qpos(self.c)
        for i in range(min(len(qpos), self.nq)):
            qposp[i] = qpos[i]

    def set_qvel(self, qvel):
        qvelp = digit_sim_qvel(self.c)
        for i in range(min(len(qvel), self.nv)):
            qvelp[i] = qvel[i]

    def hold(self):
        digit_sim_hold(self.c)

    def release(self):
        digit_sim_release(self.c)

    def apply_force(self, xfrc, body_name="digit-pelvis"):
        xfrc_array = (ctypes.c_double * 6)()
        for i in range(len(xfrc)):
            xfrc_array[i] = xfrc[i]
        digit_sim_apply_force(self.c, xfrc_array, body_name.encode())

    def sense_ground(self):
        percept = (ctypes.c_double * 6)()
        digit_sim_read_rangefinder(self.c, percept)

        ret = np.zeros(6)
        for i in range(6):
            ret[i] = percept[i]
        return ret

    def foot_force(self):
        force = np.zeros(12)
        frc_array = (ctypes.c_double * 12)()
        digit_sim_foot_forces(self.c, frc_array)
        for i in range(12):
            force[i] = frc_array[i]
        return force

    def foot_pos(self):
        pos = np.zeros(6)
        pos_array = (ctypes.c_double * 6)()
        digit_sim_foot_positions(self.c, pos_array)
        for i in range(6):
            pos[i] = pos_array[i]
        return pos

    def foot_vel(self):
        vel = np.zeros(12)
        vel_array = (ctypes.c_double * 12)()
        digit_sim_foot_velocities(self.c, vel_array)
        for i in range(12):
            vel[i] = vel_array[i]
        return vel

    def body_vel(self, body_name):
        vel = np.zeros(6)
        vel_array = (ctypes.c_double * 6)()
        digit_sim_body_vel(self.c, vel_array, body_name.encode())
        for i in range(6):
            vel[i] = vel_array[i]
        return vel

    def foot_quat(self):
        quat = np.zeros(4)
        quat_array = (ctypes.c_double * 4)()
        digit_sim_foot_quat(self.c, quat_array)
        for i in range(4):
            quat[i] = quat_array[i]
        return quat

    def clear_forces(self):
        digit_sim_clear_forces(self.c)

    def get_foot_forces(self):
        force = np.zeros(12)
        frc_array = (ctypes.c_double * 12)()
        digit_sim_foot_forces(self.c, frc_array)
        for i in range(12):
            force[i] = frc_array[i]
        lfrc = np.sqrt(np.power(force[0:3], 2).sum())
        rfrc = np.sqrt(np.power(force[6:9], 2).sum())
        return lfrc, rfrc

    def get_dof_damping(self):
        ptr = digit_sim_dof_damping(self.c)
        ret = np.zeros(self.nv)
        for i in range(self.nv):
          ret[i] = ptr[i]
        return ret

    def get_body_mass(self):
        ptr = digit_sim_body_mass(self.c)
        ret = np.zeros(self.nbody)
        for i in range(self.nbody):
          ret[i] = ptr[i]
        return ret

    def get_body_ipos(self):
        nbody = self.nbody * 3
        ptr = digit_sim_body_ipos(self.c)
        ret = np.zeros(nbody)
        for i in range(nbody):
          ret[i] = ptr[i]
        return ret

    def get_geom_friction(self):
        ptr = digit_sim_geom_friction(self.c)
        ret = np.zeros(self.ngeom * 3)
        for i in range(self.ngeom * 3):
          ret[i] = ptr[i]
        return ret

    def get_geom_rgba(self, name=None):
        if name is not None:
            ptr = digit_sim_geom_name_rgba(self.c, name.encode())
            ret = np.zeros(4)
            for i in range(4):
                ret[i] = ptr[i]
        else:
            ptr = digit_sim_geom_rgba(self.c)
            ret = np.zeros(self.ngeom * 4)
            for i in range(self.ngeom * 4):
                ret[i] = ptr[i]
        return ret


    def get_geom_quat(self):
        ptr = digit_sim_geom_quat(self.c)
        ret = np.zeros(self.ngeom * 4)
        for i in range(self.ngeom * 4):
          ret[i] = ptr[i]
        return ret

    def get_geom_pos(self, name=None):
        if name is not None:
            ptr = digit_sim_geom_name_pos(self.c, name.encode())
            ret = np.zeros(3)
            for i in range(3):
                ret[i] = ptr[i]
        else:
            ptr = digit_sim_geom_pos(self.c)
            ret = np.zeros(self.ngeom * 3)
            for i in range(self.ngeom * 3):
                ret[i] = ptr[i]
        return ret
    
    def get_geom_size(self, name=None):
        if name is not None:
            ptr = digit_sim_geom_name_size(self.c, name.encode())
            ret = np.zeros(3)
            for i in range(3):
                ret[i] = ptr[i]
        else:
            ptr = digit_sim_geom_size(self.c)
            ret = np.zeros(self.ngeom * 3)
            for i in range(self.ngeom * 3):
                ret[i] = ptr[i]
        return ret


    def set_dof_damping(self, data):
        c_arr = (ctypes.c_double * self.nv)()

        if len(data) != self.nv:
          print("SIZE MISMATCH SET_DOF_DAMPING()")
          exit(1)
        
        for i in range(self.nv):
          c_arr[i] = data[i]

        digit_sim_set_dof_damping(self.c, c_arr)

    def set_body_mass(self, data, name=None):
        # If no name is provided, set ALL body masses and assume "data" is array
        # containing masses for every body
        if name is None:
            c_arr = (ctypes.c_double * self.nbody)()

            if len(data) != self.nbody:
                print("SIZE MISMATCH SET_BODY_MASS()")
                exit(1)
            
            for i in range(self.nbody):
                c_arr[i] = data[i]

            digit_sim_set_body_mass(self.c, c_arr)
        # If name is provided, only set mass for specified body and assume
        # "data" is a single double
        else:
            digit_sim_set_body_name_mass(self.c, name.encode(), ctypes.c_double(data))

    def set_body_ipos(self, data):
        nbody = self.nbody * 3
        c_arr = (ctypes.c_double * nbody)()

        if len(data) != nbody:
          print("SIZE MISMATCH SET_BODY_IPOS()")
          exit(1)
        
        for i in range(nbody):
          c_arr[i] = data[i]

        digit_sim_set_body_ipos(self.c, c_arr)

    def set_geom_friction(self, data, name=None):
        if name is None:
            c_arr = (ctypes.c_double * (self.ngeom*3))()

            if len(data) != self.ngeom*3:
                print("SIZE MISMATCH SET_GEOM_FRICTION()")
                exit(1)

            for i in range(self.ngeom*3):
                c_arr[i] = data[i]

            digit_sim_set_geom_friction(self.c, c_arr)
        else:
            fric_array = (ctypes.c_double * 3)()
            for i in range(3):
                fric_array[i] = data[i]
            digit_sim_set_geom_name_friction(self.c, name.encode(), fric_array)

    def set_geom_rgba(self, data, name=None):
        if name is None:
                ngeom = self.ngeom * 4

                if len(data) != ngeom:
                    print("SIZE MISMATCH SET_GEOM_RGBA()")
                    exit(1)

                c_arr = (ctypes.c_float * ngeom)()

                for i in range(ngeom):
                    c_arr[i] = data[i]

                digit_sim_set_geom_rgba(self.c, c_arr)
        else:
            rgba_array = (ctypes.c_float * 4)()
            for i in range(4):
                rgba_array[i] = data[i]
            digit_sim_set_geom_name_rgba(self.c, name.encode(), rgba_array)

    def set_geom_quat(self, data, name=None):
        if name is None:
            ngeom = self.ngeom * 4

            if len(data) != ngeom:
                print("SIZE MISMATCH SET_GEOM_QUAT()")
                exit(1)

            c_arr = (ctypes.c_double * ngeom)()

            for i in range(ngeom):
                c_arr[i] = data[i]

            digit_sim_set_geom_quat(self.c, c_arr)
        else:
            quat_array = (ctypes.c_double * 4)()
            for i in range(4):
                quat_array[i] = data[i]
            digit_sim_set_geom_name_quat(self.c, name.encode(), quat_array)
    
    def set_geom_pos(self, data, name=None):
        if name is None:
            ngeom = self.ngeom * 3

            if len(data) != ngeom:
                print("SIZE MISMATCH SET_GEOM_POS()")
                exit(1)

            c_arr = (ctypes.c_double * ngeom)()

            for i in range(ngeom):
                c_arr[i] = data[i]

            digit_sim_set_geom_pos(self.c, c_arr)
        else:
            array = (ctypes.c_double * 3)()
            for i in range(3):
                array[i] = data[i]
            digit_sim_set_geom_name_pos(self.c, name.encode(), array)

    def set_geom_size(self, data, name=None):
        if name is None:
            ngeom = self.ngeom * 3

            if len(data) != ngeom:
                print("SIZE MISMATCH SET_GEOM_POS()")
                exit(1)

            c_arr = (ctypes.c_double * ngeom)()

            for i in range(ngeom):
                c_arr[i] = data[i]

            digit_sim_set_geom_size(self.c, c_arr)
        else:
            array = (ctypes.c_double * 3)()
            for i in range(3):
                array[i] = data[i]
            digit_sim_set_geom_name_size(self.c, name.encode(), array)

    def set_const(self):
        digit_sim_set_const(self.c)

    def full_reset(self):
        digit_sim_full_reset(self.c)

    def get_hfield_nrow(self):
        return digit_sim_get_hfield_nrow(self.c)

    def get_hfield_ncol(self):
        return digit_sim_get_hfield_ncol(self.c)

    def get_nhfielddata(self):
        return digit_sim_get_nhfielddata(self.c)

    def get_hfield_size(self):
        ret = np.zeros(4)
        ptr = digit_sim_get_hfield_size(self.c)
        for i in range(4):
            ret[i] = ptr[i]
        return ret

    # Note that data has to be a flattened array. If flattening 2d numpy array, rows are y axis
    # and cols are x axis. The data must also be normalized to (0-1)
    def set_hfield_data(self, data):
        nhfielddata = self.get_nhfielddata()
        if len(data) != nhfielddata:
            print("SIZE MISMATCH SET_HFIELD_DATA")
            exit(1)
        data_arr = (ctypes.c_float * nhfielddata)(*data)
        digit_sim_set_hfielddata(self.c, ctypes.cast(data_arr, ctypes.POINTER(ctypes.c_float)))
    
    def get_hfield_data(self):
        nhfielddata = self.get_nhfielddata()
        ret = np.zeros(nhfielddata)
        ptr = digit_sim_hfielddata(self.c)
        for i in range(nhfielddata):
            ret[i] = ptr[i]
        return ret

    def set_hfield_size(self, data):
        if len(data) != 4:
            print("SIZE MISMATCH SET_HFIELD_SIZE")
            exit(1)
        size_array = (ctypes.c_double * 4)()
        for i in range(4):
            size_array[i] = data[i]
        digit_sim_set_hfield_size(self.c, size_array)

    def __del__(self):
        digit_sim_free(self.c)

class DigitVis:
    def __init__(self, c, modelfile):
        self.v = digit_vis_init(c.c, modelfile.encode('utf-8'))

    def draw(self, c):
        state = digit_vis_draw(self.v, c.c)
        return state

    def valid(self):
        return digit_vis_valid(self.v)

    def ispaused(self):
        return digit_vis_paused(self.v)

    # Applies the inputted force to the inputted body. "xfrc_apply" should contain the force/torque to 
    # apply in Cartesian coords as a 6-long array (first 3 are force, last 3 are torque). "body_name" 
    # should be a string matching a body name in the XML file. If "body_name" doesn't match an existing
    # body name, then no force will be applied. 
    def apply_force(self, xfrc_apply, body_name):
        xfrc_array = (ctypes.c_double * 6)()
        for i in range(len(xfrc_apply)):
            xfrc_array[i] = xfrc_apply[i]
        digit_vis_apply_force(self.v, xfrc_array, body_name.encode())

    def reset(self, c):
        digit_vis_full_reset(self.v, c.c)

    def set_cam(self, body_name, zoom, azimuth, elevation):
        digit_vis_set_cam(self.v, body_name.encode(), zoom, azimuth, elevation)

    def __del__(self):
        digit_vis_free(self.v)


if __name__ == '__main__':
    sim = DigitSim("../model/digit.xml")
    vis = DigitVis(sim, "../model/digit.xml")
    u = pd_in_t()
    P = [100, 100, 300, 300, 50, 50,
         100, 100, 300, 300, 50, 50,
         50, 50, 50, 50,
         50, 50, 50, 50]
    D = [10, 10, 20, 20, 5, 5,
         10, 10, 20, 20, 5, 5,
         5, 5, 5, 5,
         5, 5, 5, 5]
    P = [p_i / 1.5 for p_i in P]
    D = [d_i / 1.5 for d_i in D]
    for i in range(20):
      u.pGain[i] = P[i]
      u.dGain[i] = D[i]
      u.pTarget[i] = 0
      u.dTarget[i] = 0
      u.torque[i] = 0
    for i in range(100000):
      s = sim.step_pd(u)
      #print(['{:3.1f}'.format(x) for x in s.motor_pos[:]])
      if i % 60 == 0:
        vis.draw(sim)
    print("Done.")
