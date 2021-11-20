import numpy as np
import random
from util.quat import euler2quat

class Stairs:
    """Class used to introduce and randomize stair objects in the real world.
       Used by Cassie and Digit to reduce spaghetti-ness. Expects the MuJoCo
       .xml file to have box geoms titled 'box1', ... 'boxn' to be used for
       stair generation.

       Constructor expects
        'self.sim.set_geom_size'
        'self.sim.get_geom_size'
        'self.sim.set_geom_pos'
        'self.sim.get_geom_pos'
    """

    def __init__(self, sim, geom_names):
        self.available_geoms = geom_names

        if not hasattr(sim, 'set_geom_size'):
            raise RuntimeError

        if not hasattr(sim, 'get_geom_size'):
            raise RuntimeError

        if not hasattr(sim, 'set_geom_pos'):
            raise RuntimeError

        if not hasattr(sim, 'get_geom_pos'):
            raise RuntimeError

        self.sim = sim

        self.rise_noise  = [-0.02, 0.02]
        self.run_noise   = [-0.05, 0.05]
        self.roll_noise = [-0.00, 0.0]
        self.pitch_noise = [-0.15, 0.15]
        self.stair_width = 10

    def _create_step(self, box, start_x, start_y, start_z, length, rise, randomize_slope=False, slat=False):
        stair_rise   = rise + np.random.uniform(*self.rise_noise)
        stair_length = length + np.random.uniform(*self.run_noise)

        x = start_x + stair_length/2
        z = start_z

        vert_size = 0.01 if slat else np.abs(stair_rise)/2
        if np.sign(rise) < 0:
            vert_pos  = z + rise if slat else z + stair_rise/2
        else:
            vert_pos  = z if slat else z + stair_rise/2

        self.sim.set_geom_size([stair_length/2, self.stair_width, vert_size], box)
        self.sim.set_geom_pos([x, start_y, vert_pos], box)
        
        if randomize_slope:
            geom_plane = [np.random.uniform(*self.roll_noise), np.random.uniform(*self.pitch_noise), 0]
            quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            self.sim.set_geom_quat(quat_plane, box)
        
        return x + stair_length/2, z + stair_rise

    def create_stairs(self, start_x, start_y, start_z, default_length, default_rise, height, slats=False, quat=False):
        height = height-1 if height % 2 != 0 else height
        max_boxes = len(self.available_geoms)

        boxes = int(min(height*2, max_boxes-1))
            
        current_x = start_x
        current_y = start_y
        current_z = start_z
        stair_width = 10
        midway = boxes // 2

        for box in self.available_geoms[:midway]:
            current_x, current_z = self._create_step(box, current_x, current_y, current_z, default_length, default_rise, slat=slats, randomize_slope=quat)

        z_before = current_z
        intermediate = np.random.uniform(0.5, 2)
        current_x, current_z = self._create_step(self.available_geoms[midway], current_x, current_y, current_z, intermediate, default_rise, slat=slats, randomize_slope=quat)
        self.exempt_stairs = [midway]
        current_z = z_before

        for box in self.available_geoms[midway+1:boxes+1]:
            current_x, current_z = self._create_step(box, current_x, current_y, current_z, default_length, -default_rise, slat=slats, randomize_slope=quat)

        self.bounds = [start_x - default_length/2, current_x + default_length/2]

        for box in self.available_geoms[boxes+1:]:
            current_x += default_length/2
            self.sim.set_geom_size([0.1, 0.1, 0.1], box)
            self.sim.set_geom_pos([current_x, 20, 0], box)

    def check_foot_step(self, x, y, z):
        # default foot position is mid-foot, so check heel and toe positions.
        # should probably rotate offsets to account for foot orientation but it's likely
        # not a huge deal
        heel_step, heel_z = self.check_step(*(np.array((x, y, 0)) - [0.06, 0, 0]))
        toe_step, toe_z   = self.check_step(*(np.array((x, y, 0)) + [0.10, 0, 0]))

        if heel_step != toe_step and (heel_step not in self.exempt_stairs and toe_step not in self.exempt_stairs):
            caught_edge = True
        else:
            caught_edge = False

        return [heel_step, toe_step][np.argmax([heel_z, toe_z])], caught_edge

    def check_step(self, x, y, z):
        ret_z   = z
        box_num = None
        for i, box in enumerate(self.available_geoms):
              box_d, box_w, box_h = self.sim.get_geom_size(name=box)
              box_x, box_y, box_z = self.sim.get_geom_pos(name=box)

              if x < box_x + box_d and x > box_x - box_d and \
                  y < box_y + box_w and y > box_y - box_w:

                  ret_z = box_z + box_h
                  box_num = i
        return box_num, ret_z

