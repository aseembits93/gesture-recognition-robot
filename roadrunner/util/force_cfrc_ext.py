"""
In MuJoCo 2.0, the function mj_rnePostConstraint is no longer
called by default. This function calculates various quantities,
among them cfrc_ext, which we need to use for research purposes.
Adding a force sensor to the .xml forces MuJoCo to compute the
desired function, but editing the .xml files by hand is tedious.

This script adds a dummy force sensor to gym environment xml files
to compell MuJoCo to calculate cfrc_ext for these nevironments.
"""

import os
import gym

if __name__ == '__main__':
  gym_paths = gym.__path__

  for gym_path in gym_paths:

    asset_path = os.path.join(str(gym_path), 'envs/mujoco/assets')
    force_calc = ''.join(["<!--</worldbody>-->\n",
                          "<!-- THIS SECTION ADDED BY SCRIPT TO FORCE CFRC_EXT CALCULATION -->\n",
                          "<site name='dummysite'/>\n",
                          "</worldbody>\n",
                          "<sensor>\n",
                          "<force name='this_forces_mujoco_to_calculate_cfrc_ext' site='dummysite'/>",
                          "\n</sensor>\n",
                          "<!-- END MODIFIED SECTION -->\n"])
    envs = ['hopper', 'humanoid', 'walker2d', 'half_cheetah', 'ant']

    for env in envs:
        worldbody_string = '</worldbody>'
        path = os.path.join(asset_path, env + '.xml')
        with open(path, 'r+') as f:
            text = f.read()
            if force_calc in text:
                print('{} already modified, skipping.'.format(path))
                continue 
            idx = text.find(worldbody_string)

        new_text = text[:idx] + force_calc + text[idx + len(worldbody_string)+1:]
        with open(path, 'w') as f:
          f.write(new_text)
          print('Wrote to {} successfully.'.format(path))
