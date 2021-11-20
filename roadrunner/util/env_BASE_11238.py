import os
import time
import torch
import numpy as np

def env_factory(**kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.
    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized,
    this allows us to pass their constructors to Ray remote functions instead
    (since the gym registry isn't shared across ray subprocesses we can't simply
    pass gym.make() either)
    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    path = kwargs['env']
    if 'dummyplanner' in path.lower():
        from envs.meta.dummy_planner import DummyPlannerEnv

        return partial(DummyPlannerEnv, **kwargs)

    if 'dummy' in path.lower():
        from envs.meta.dummy_planner import DummyEnv

        return partial(DummyEnv, **kwargs)

    if 'dynconfig' in path.lower():
        from envs.meta.dynamic_config_planner import DynamicConfigPlanner

        return partial(DynamicConfigPlanner, **kwargs)

    if 'waypoint' in path.lower():
        from envs.meta.waypoint_planner import WaypointPlanner

        return partial(WaypointPlanner, **kwargs)

    if 'gaitplanner_map' in path.lower():
        from envs.meta.gait_planner_map import GaitPlannerMapEnv

        return partial(GaitPlannerMapEnv, **kwargs)

    if 'gaitplanner' in path.lower():
        # from envs.meta.gait_planner_map import GaitPlannerMapEnv
        # return partial(GaitPlannerMapEnv, **kwargs)

        from envs.meta.gait_planner import GaitPlannerEnv
        
        return partial(GaitPlannerEnv, **kwargs)

    if 'cassiestandup' in path.lower():
        from envs.envs import CassieStandupEnv
        return partial(CassieStandupEnv, **kwargs)

    if 'cassie' in path.lower():
        from envs.envs import CassieLocomotionEnv
        if True:
            print("Created cassie env with arguments:")
            print("\tsimrate:                {}".format(kwargs['simrate']))
            #print("\treward:                 {}".format(kwargs['reward']))
            print("\tdynamics randomization: {}".format(kwargs['dynamics_randomization']))
            print("\timpedance control:      {}".format(kwargs['impedance']))
            #print("\tincentives:             {}".format(kwargs['incentive']))
            print("\tstanding mode:          {}".format(kwargs['standing']))
            print("\tfixed gait (hopping):   {}".format(kwargs['fixed_hop']))
            print("\tfixed gait (walking):   {}".format(kwargs['fixed_walk']))
            print("\tphase transition std:   {}".format(kwargs['phase_std']))
            print("\theight control:         {}".format(kwargs['height']))
            print("\tstairs:                 {}".format(kwargs['stairs']))
            print("\tperception:             {}".format(kwargs['perception']))
            #print("\ttask:                   {}".format(kwargs['task']))
            #print("\tgaze control:           {}".format(kwargs['gaze_control']))
            # print("\tinput_turn_rate:        {}".format(kwargs['input_turn_rate']))

        return partial(CassieLocomotionEnv, **kwargs)

    if 'digit' in path.lower():
        from envs.digit.digit import DigitEnv
        if True:
            print("Created digit env with arguments:")
            print("\tsimrate:                {}".format(kwargs['simrate']))
            #print("\treward:                 {}".format(kwargs['reward']))
            print("\tdynamics randomization: {}".format(kwargs['dynamics_randomization']))
            print("\timpedance control:      {}".format(kwargs['impedance']))
            print("\tincentives:             {}".format(kwargs['incentive']))
            print("\tstanding mode:          {}".format(kwargs['standing']))
            print("\tfixed gait (hopping):   {}".format(kwargs['fixed_hop']))
            print("\tfixed gait (walking):   {}".format(kwargs['fixed_walk']))
            print("\tphase transition std:   {}".format(kwargs['phase_std']))
            print("\theight control:         {}".format(kwargs['height']))
            print("\tstairs:                 {}".format(kwargs['stairs']))
            print("\tperception:             {}".format(kwargs['perception']))
            print("\ttask:                   {}".format(kwargs['task']))

        return partial(DigitEnv, **kwargs)

    elif 'periodic' in path.lower():
        if 'hopper' in path.lower():
            from envs.periodic_hopper import PeriodicHopperEnv
            env = PeriodicHopperEnv

        elif 'walker2d' in path.lower():
            from envs.periodic_walker2d import PeriodicWalker2dEnv
            env = PeriodicWalker2dEnv

        elif 'humanoid' in path.lower():
            from envs.periodic_humanoid import PeriodicHumanoidEnv
            env = PeriodicHumanoidEnv

        elif 'ant' in path.lower():
            from envs.periodic_ant import PeriodicAntEnv
            env = PeriodicAntEnv

        if 'noclock' in path.lower():
            orientation_only = True
        else:
            orientation_only = False

        if 'clockonly' in path.lower():
            periodic_only = True
        else:
            periodic_only = False

        if 'skip' in path.lower():
            skipping = True
        else:
            skipping = False

        return partial(env, skipping=skipping, periodic_only=periodic_only, orientation_only=orientation_only)

    else:
        import gym
        spec = gym.envs.registry.spec(path)
        _kwargs = spec._kwargs.copy()
        _kwargs.update(kwargs)

        try:
            if callable(spec._entry_point):
                cls = spec._entry_point(**_kwargs)
            else:
                cls = gym.envs.registration.load(spec._entry_point)
        except AttributeError:
            if callable(spec.entry_point):
                cls = spec.entry_point(**_kwargs)
            else:
                cls = gym.envs.registration.load(spec.entry_point)

        return partial(cls, **_kwargs)

def train_normalizer(env_fn, policy, min_timesteps, max_traj_len=1000, noise=0.5):
    with torch.no_grad():
        env = env_fn()
        env.dynamics_randomization = False

        total_t = 0
        while total_t < min_timesteps:
            state = env.reset()
            done = False
            timesteps = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            while not done and timesteps < max_traj_len:
                if noise is None:
                    action = policy.forward(state, update_norm=True, deterministic=False).numpy()
                else:
                    action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
                state, _, done, _ = env.step(action)
                # env.render()
                timesteps += 1
                total_t += 1

def hrl_train_normalizer(env_fn, model_tree, min_timesteps, max_traj_len=1000, noise=0.5):
    with torch.no_grad():
        env = env_fn()
        env.dynamics_randomization = False

        total_t = 0
        while total_t < min_timesteps:
            state = env.reset()
            done = False
            timesteps = 0

            if model_tree.is_recurrent:
                model_tree.init_hidden_state()

            while not done and timesteps < max_traj_len:
                if noise is None:
                    action = model_tree(state, deterministic=False, update_norm=True).numpy()
                else:
                    action = model_tree.noisy_forward(state, noise, deterministic=False, update_norm=True).numpy()
                state, _, done, _ = env.step(action)
                # env.render()
                timesteps += 1
                total_t += 1
