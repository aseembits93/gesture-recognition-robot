def _dfn(d):
    """helper for debugging planner envs
    """
    return ''.join(['\t' for _ in range(d)])

def _flatten_sample(sample, depth=1):
    """helper for Planner.step()
    """
    if sample is None:
        return sample
    elif sample[0] is None:
        return sample[0]
    else:
        states, actions, rewards, dones, subsample = sample
        states    = np.vstack(states)
        actions   = np.vstack(actions)
        rewards   = np.vstack(rewards)
        dones     = np.vstack(dones).flatten()
        subsample = _flatten_sample(subsample, depth=depth+1)
        return states, actions, rewards, dones, subsample

class PlannerEnv:
    def __init__(self, subenv_fn_stack, evaluation_mode=False, interactive_mode=False, **kwargs):

        self.super_commands = self.get_super_interface() # commands from further up the chain

        # Default Settings
        self.has_superenv = False
        self.can_mirror = False
        self.evaluation_mode = evaluation_mode
        self.interactive_mode = interactive_mode

        # Assumes that last env in stack specifies the PD_Env
        self.has_pd_env = True if len(subenv_fn_stack) == 1 else False

        if not self.has_pd_env:
            self.subenv = subenv_fn_stack[0](subenv_fn_stack[1:], **kwargs)
            self.sub_commands   = self.get_sub_interface()   # commands to send further down the chain
            self.subenv.has_superenv = True
        else:
            self.sub_commands = None
            self.subenv = subenv_fn_stack[0]()  # This instantiates the PD_Env

        self.has_periodic_env = hasattr(self.subenv, 'periodic_reward')

        # Places to keep information across planners
        self.cmd_dict = dict()                 # info from set_command()
        self.shared_info = dict()              # info like speed bounds, rendering information

    def randomize_super_command(self, calling_rate=1, init=False, **kwargs):
        raise NotImplementedError

    def get_super_interface(self, **kwargs):
        """
            Returns set of keys for commands received from superenv
        """
        raise NotImplementedError

    def get_sub_interface(self, **kwargs):
        """
            Returns set of keys for commands sent to subenv
        """
        return self.subenv.get_super_interface()

    def reset_info(self):
        """
            Resets the planner local attributes.
        """
        pass

    def get_state(self):
        """
            Returns the state from this environment for the policy. Called after subenvs have stepped.
        """
        raise NotImplementedError

    def compute_reward(self, action):
        """
            Reward for current level of hierarchy after action has been enacted
        """
        raise NotImplementedError

    def compute_done(self):
        """
            Whether or not to terminate episode after last action was enacted
            This is only called for the lowest-level planner whose self.has_pd_env == True
            i.e. if this isn't implemented for a planner, that planner cannot be at bottom of hierarchy
        """
        raise NotImplementedError

    def make_command(self, vec):
        """
            Does "work" at this level of hierarchy.
            Expected to output a dict representing the command for the subenv
        """
        raise NotImplementedError

    def set_command(self, cmd):
        """
            Set command
        """
        for c in cmd:
            if c not in self.super_commands:
                print("ERROR: planner sent unexpected command '{}'.".format(c))
                raise RuntimeError
        for c in self.super_commands:
            if c not in cmd:
                print("ERROR: planner did not send expected command '{}'.".format(c))
                raise RuntimeError
        
        for c in cmd:
            self.cmd_dict[c] = cmd[c]

    def get_info(self, attribute_name, *args, **kwargs):
        """
            Fetch info from robot or shared_info dict
        """
        if attribute_name in self.shared_info:
            return self.shared_info[attribute_name]()
        else:
            return self.subenv.get_info(attribute_name, *args, **kwargs)
        
        print("ERROR: planner asked for shared info that doesn't exist '{}'".format(attribute_name))

    def get_robot_info(self, attribute_name):
        """
            Get info from robot
        """
        return self.subenv.get_info(attribute_name)

    def set_info(self, attribute_name, info):
        """
            Set info in this level's shared_info dict
        """
        self.shared_info[attribute_name] = info

    def set_robot_info(self, attribute_name, *args, **kwargs):
        """
            Set info in robot.
        """
        return self.subenv.set_robot_info(attribute_name, *args, **kwargs)

    def print_command(self):
        """
            Prints self.cmd_info dict. Useful for debugging
        """
        print(json.dumps(self.cmd_info, sort_keys=True, indent=2, default=str))

    def print_info(self):
        """
            Prints self.info dict. Useful for debugging
        """
        print(json.dumps(self.info, sort_keys=True, indent=2, default=str))

    def reset(self):
        """
            Reset environment at this level of hierarchy
        """
        # NOTE: bottom-most pd_env will return nothing
        # NOTE: this used to be at bottom of this function
        if not self.has_pd_env:
            self.last_substate = self.subenv.reset()
        else:
            self.subenv.reset()

        # If this is the top-level planner, generate commands randomly.
        self.set_command(self.randomize_super_command(init=True))
        self.reset_info()

        return self.get_state()

    def prestep(self):
        """
            PeriodicEnv needs to update internal variables based on cmd_dict before reward 
            calculation takes place in step(), so that should be done in prestep().
        """
        pass

    def update_trackers(self, action):
        pass

    def step(self, action, policies, critics=None, deterministic=[True], render=False, update_norm=False, depth=1, mirror=False, calling_rate=1, keypress=None):
        """
            Step command used by all planner environments. Recursively returns
            state/return information from lower levels as the fourth return value.
        """

        if not self.has_superenv: # maybe pass an action_repeat int for this fcn
            if not self.interactive_mode:
                self.set_command(self.randomize_super_command(calling_rate=calling_rate))
            else:
                self.set_command(self.interactive_control(keypress))
        self.prestep()

        # Enact action, get command for next level
        policy_command = self.make_command(action)

        if self.has_pd_env:
            # PD steps at bottom of hierarchy
            for i in range(self.simrate):
                self.subenv.step(action)

            # This was simple way to let PD_Env objects finish out their robot tracking info
            self.subenv.update_trackers()

            if render:
                self.subenv.render()

            subdones  = []
            subsample = None

        else:
            # Set command for next level
            self.subenv.set_command(policy_command)

            # Recursive steps through hierarchy
            substates, subrewards, subdones, subactions = [], [], [], []
            subsubsamples = [], [], [], [], []

            num_substeps = 1
            while True: # TODO Come up with a better way of handling periodic steps, aperiodic steps, pd steps in this function
                if mirror and self.subenv.can_mirror:
                    state = self.subenv.mirror_state(self.last_substate)
                else:
                    state = self.last_substate

                # Get action for subenv
                a = policies[0](state, deterministic=deterministic[0], update_norm=update_norm).numpy()

                if mirror and self.subenv.can_mirror:
                    a = self.subenv.mirror_action(a)

                substates  += [state]
                subactions += [a]

                # recursive step
                sample = self.subenv.step(a, policies[1:], render=render, 
                                                           depth=depth+1, 
                                                           mirror=mirror, 
                                                           calling_rate=calling_rate//self.planner_rate, 
                                                           deterministic=deterministic[1:],
                                                           update_norm=update_norm)
                if len(sample) == 4:
                    next_state, reward, done, subsubsample = sample  # gym-style envs don't return action
                elif len(sample) == 5:
                    next_state, _, reward, done, subsubsample = sample  # h-envs return action, which we already have

                subrewards += [reward]
                subdones   += [done]

                if subsubsample is not None and len(subsubsample) > 0:
                    subsubsamples = [x + [y] for x,y in zip(subsubsamples, subsubsample)]
                else:
                    subsubsamples = None

                # save the subenv's new state so we can continue computing new actions next time step() is called
                self.last_substate = next_state
                num_substeps += 1
                if done:
                    break

                if self.has_periodic_env:
                    current_phase_offset = self.subenv.phase % (self.subenv.phase_len // self.planner_rate)
                    next_phase_offset = (self.subenv.phase + self.subenv.phase_add) % (self.subenv.phase_len // self.planner_rate)

                    if current_phase_offset > next_phase_offset:
                        break
                elif num_substeps > self.planner_rate:
                    break
            subsample  = (substates, subactions, subrewards, subdones, subsubsamples)

        # After done with lower levels, finish step for current level
        # Inside the following functions you can use self.get_robot_info() to get robot info from PD_Env, which is no longer stepping
        subsamples = _flatten_sample(subsample)
        supdone    = self.compute_done()
        supreward  = self.compute_reward(action)

        self.update_trackers(action)
        next_state = self.get_state()

        return next_state, action, supreward, True in subdones or supdone, (subsamples)

    def add_render_obj(self, obj_id, pos, size, rgba, orient=None):
        return self.subenv.add_render_obj(obj_id, pos, size, rgba, orient)

    def remove_render_obj(self, obj_id):
        return self.subenv.remove_render_obj(obj_id)

    def clear_render_objs(self):
        return self.subenv.clear_render_objs()

    def update_render_obj(self, obj_id, pos=None, size=None, rgba=None, orient=None):
        return self.subenv.update_render_obj(obj_id, pos=pos, size=size, rgba=rgba, orient=orient)


    def log_cmd(self):
        """
            Return a string with logging info for incoming command at this level of env. Used in interactive eval
        """
        raise NotImplementedError

    def interactive_control(self, c):
        """
            Given a keypress resulting in 'char', change commands or other info in environment for interactive user control
        """
        raise NotImplementedError


