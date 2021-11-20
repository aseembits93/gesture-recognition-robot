import numpy as np
import random

from util.clock import Phase, vonmises_func
from collections import namedtuple

_imported_matplotlib = False

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

class Behavior:
    """Template class used to describe a single behavior (sequences of phases mapped 
       to measured quantities in a phase-based reward function). A behavior is defined 
       in the phase function framework as a list of several measurable quantities, with 
       a sequence of probabilistic phases mapped to each quantity.
       The corresponding reward function for this behavior is the sum of the 
       measured quantity * the value of the probabilistic phase function.

    Attributes:
        phase_dict (dict): Dictionary with key-value pairs mapping a measured 
        quantity in the reward (str) to a sequence of phases

    Note:
        ratio and period_shift are optional and non-essential, but will 
        be used if the plot method is called on the Behavior object
    """
    def __init__(self, ratio, period_shift, phase_dict=dict(), phase_len=1500):
        # Behavior info
        self.phase_dict   = phase_dict
        self.ratio        = ratio
        self.period_shift = period_shift
        self.phase_len    = phase_len

        # Domain
        self.xlim = 1.0
        self.x = np.linspace(0, self.xlim, num=phase_len)

        # Precomputed reward components
        self.reward_components = self._reward_components()

    def expected_coeffs(self, name, phase):
        """Returns the expected phase coefficient given a name of a component.
        """
        return self.reward_components[name][phase]

    def reward(self, phase, q, w, b=1, debug=False):
        """Returns the total reward given a phase index phase and a dictionary
           of named measurements.

        Attributes:
            phase (int):         integer index within the domain self.x
            q (dict):            dictionary mapping a quantity name (str) 
                                 to a quantity (str).
            w (dict):            dictionary mapping relative weightings
                                 to measurement names.
        """
        cost = 0
        for name in self.reward_components.keys():
            cost += w[name] * q[name] * self.reward_components[name][phase]

        if debug:
            print("total reward {:4.3f}".format(b - cost))
            print("derived from: {:3.2f} - {:4.3f}:".format(b, cost))
            for name in self.reward_components.keys():
                print("\t{:20s}: {:5.3f} * {:6.3f} * {:6.3f} = {:6.3f}".format(name, w[name], q[name], self.reward_components[name][phase], w[name]*q[name]*self.reward_components[name][phase]))
            
        return b - cost

    def _reward_components(self):
        """Returns a dictionary of each reward component mapped to its 
           registered quantity name.
        """
        reward_components = {}
        for name, phase_info in self.phase_dict.items():
            if phase_info is None:
                reward_components[name] = -np.ones(self.phase_len)
            else:
                seq, shift_group = phase_info
                reward_components[name] = vonmises_func(self.x, seq, shift=self.period_shift[shift_group])
        return reward_components

    def plot(self, plot_name=""):
        """Plot the behavior's decomposed components
        """
        if not _imported_matplotlib:
            import matplotlib.pyplot as plt
        reward_components = self._reward_components()
        self.fig, self.axs = plt.subplots(nrows=len(reward_components), ncols=1, sharex=True, sharey=True, figsize=(10, 15))
        for i, (name, component) in enumerate(reward_components.items()):
            self.axs[i].plot(self.x, component)
            self.axs[i].set_ylabel(name)
        if self.ratio is not None and self.period_shift is not None:
            self.axs[0].set_title(f"{plot_name} (ratio= {list(map(prettyfloat, self.ratio))}, shift= {list(map(prettyfloat, self.period_shift))})")
        else:
            self.axs[0].set_title(plot_name)


class PeriodicEnv:
    """Class used to describe temporal variations in behaviors and their associated phase-based reward function.

    A behavior space describes all of the variations of a behavior within some specified constraint.
    For example, a behavior space for walking specifies all of the possible walking behaviors

    Attributes:
        ratio_func (function): input-less function which when called gives a possible list of ratios for the behavior space
        shift_func (function): input-less function which when called gives a possible shift
        std (float): optional global std for phases in the Behavior objects created by this class (default is 0.2)
        phaselen (int): optional global phase_len for Behavior objects created by this class (default is 1700)
        schema (dict): optional dictionary mapping quantity names to sequences of phases.

    """

    def __init__(self, num_phases, num_groups, std=0.2, phase_len=1700, phase_add=50):
        self.std        = std
        self.phase_len  = phase_len
        self.phase      = 0
        self.schema     = dict()
        self.num_phases = num_phases
        self.num_groups = num_groups

    def randomize_ratio(self):
        raise NotImplementedError

    def randomize_period_shift(self):
        raise NotImplementedError

    def randomize_phase(self):
        #self.phase = np.random.randint(0, self.phase_len)
        self.phase = 0

    def update_phase(self):
        self.phase += self.phase_add

        if self.phase >= self.phase_len:
            self.phase = self.phase % (self.phase_len - 1)

    def add_reward_component(self, name, group=None, coeff=None):
        """add a new component (quantity and associated phases) to the behavior space description
        """
        if coeff is not None and group is not None:
            assert(len(coeff) == self.num_phases)
            self.schema[name] = (coeff, group)
        else:
            self.schema[name] = None

    def _precompute_behavior(self, ratio, period_shift, phase_len, std):
        """helper method for creating an exact behavior from specified ratios and shift,
        """
        # construct behavior
        phase_dict = {}
        for name, component in self.schema.items():
            if component is None:
                phase_dict[name] = None
            else:
                time, out = 0, []
                for r, coeff in zip(ratio, component[0]):
                    out.append(Phase(start=time, end=time+r, std=std, coeff=coeff))
                    time += r
                phase_dict[name] = out, component[1]

        self.behavior = Behavior(ratio=ratio, period_shift=period_shift, phase_dict=phase_dict, phase_len=phase_len)

    def sample_behavior(self, sample_ratio=True, sample_shift=True):
        """sample a behavior from the space of behaviors.
        """
        # sample ratios
        if sample_ratio:
            sample_ratios = self.randomize_ratio()
        else:
            sample_ratios = self.behavior.ratio

        # sample shift
        if sample_shift:
            sample_shifts = self.randomize_period_shift()
        else:
            sample_shifts = self.behavior.period_shift

        self._precompute_behavior(sample_ratios, sample_shifts, self.phase_len, self.std)

    def set_behavior(self, ratio, period_shift):
        """set a behavior from given ratios and shifts
        """

        self._precompute_behavior(ratio, period_shift, self.phase_len, self.std)

    def periodic_reward(self, q, w, b=1, debug=False):
        return self.behavior.reward(self.phase, q, w, b=b, debug=debug)

    def input_clocks(self):
        clock = [np.sin(2 * np.pi * ((self.phase/self.phase_len)+s)) for s in self.behavior.period_shift]
        return clock

    def behavior_ratios(self):
        return [r for r in self.behavior.ratio]


