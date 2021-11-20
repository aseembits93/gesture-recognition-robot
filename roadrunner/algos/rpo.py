"""Proximal Policy Optimization (clip objective)."""
import sys
import os
import ray
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from time import time

from algos.ppo import Buffer, merge_buffers, PPO_Optim
from nn.actor import *
from nn.critic import *
from nn.hierarchy import *
from envs.meta.gait_planner import *
from envs.planner import dfn

from util.env import *

class RPO_Worker:
    """
        Generic template for a worker (sampler or optimizer) for RPO

        Args:
            actor_stack (list): list of actor pytorch network
            critic_stack (list): list of critic pytorch network

        Attributes:
            system: a RecursiveSystem object
    """
    def __init__(self, actor_stack, critic_stack, env_fn_stack, fixed_pol):
        actor_stack = [deepcopy(a) for a in actor_stack]
        critic_stack = [deepcopy(c) for c in critic_stack]
        self.system = RecursiveSystem(actor_stack, critic_stack, env_fn_stack, fixed_pol)

    def sync_policies(self, new_actor_params_list, new_critic_params_list, input_norms=None):
        """
        Function to sync the actor and critic parameters with new parameters.

        Args:
            new_actor_params_list (list): list of torch dictionaries of new actor parameters to copy over
            new_critic_params_list (list): list of torch dictionaries of new critic parameters to copy over
            input_norms (list): list of ints of Running counter of states for normalization 
        """
        new_actor_params_list = ray.get(new_actor_params_list)
        new_critic_params_list = ray.get(new_critic_params_list)
        for actor, new_actor_params in zip(self.system.policy_stack, new_actor_params_list):
            for p, new_p in zip(actor.parameters(), new_actor_params):
                p.data.copy_(new_p)

        for critic, new_critic_params in zip(self.system.critic_stack, new_critic_params_list):
            for p, new_p in zip(critic.parameters(), new_critic_params):
                p.data.copy_(new_p)

        if input_norms is not None:
            for i, input_norm in zip(range(len(self.system.policy_stack)), input_norms):
                self.system.policy_stack[i].welford_state_mean, 
                self.system.policy_stack[i].welford_state_mean_diff, 
                self.system.policy_stack[i].welford_state_n = input_norm

                self.system.critic_stack[i].copy_normalizer_stats(self.system.policy_stack[i])

@ray.remote
class RPO_Sampler(RPO_Worker):
    """
    Worker for sampling experience for RPO

    Args:
        actor_stack (list): list of actor pytorch network
        critic_stack (list): list of critic pytorch network
        env_fn_stack (list): list of environment constructor function
        fixed_pol (list): list of flags for which levels in hierarchy are fixed
        gamma (list): discount factors across stack
        dynamics_randomization (bool): if dynamics_randomization is enabled in environment
        seed: random seed

    Attributes:
        system: RecursiveSystem object
        gamma: discount factor
        dynamics_randomization: if dynamics_randomization is enabled in environment
    """
    def __init__(self, actor_stack, critic_stack, env_fn_stack, fixed_pol, gamma=[0.95], dynamics_randomization=False, seed=None):
        self.dynamics_randomization = dynamics_randomization
        self.gamma = gamma
        RPO_Worker.__init__(self, actor_stack, critic_stack, env_fn_stack, fixed_pol)

    def evaluate(self, trajs=1, max_traj_len=300):
        """
        Function to evaluate

        Args:
            max_traj_len: maximum trajectory length of an episode
            trajs: minimum trajectories to evaluate for
        """
        return self.system.eval(trials=trajs, max_len=max_traj_len)

    def collect_experience(self, min_steps=50, max_traj_len=30):
        """
        Function to sample experience

        Args:
            max_traj_len: maximum trajectory length of an episode
            min_steps: minimum total steps to sample
        """
        torch.set_num_threads(1)
        with torch.no_grad():
            num_steps = 0
            memories = [Buffer(g) for g in self.gamma]

            while num_steps < min_steps:
                self.system.reset()
                done = False
                traj_len = 0

                while not done and traj_len < max_traj_len:
                    recursive_sample = self.system.step(deterministic=False, render=False)
                    #traj_len  += self.system.rate
                    #num_steps += self.system.rate

                    tmp_sample = recursive_sample
                    for _ in memories[1:]:
                        _, _, _, _, tmp_sample = recursive_sample
                    traj_len += len(tmp_sample[0])
                    num_steps += len(tmp_sample[0])


                    for depth, (buff, critic) in enumerate(zip(memories, self.system.critic_stack)):
                        states, actions, rewards, dones, next_level = recursive_sample 

                        # dones will propagate up the stack through recursion, but must also propagate down
                        # correctly so that the terminal values are calculated correctly. So, if locomotion_planner
                        # sets done=True, then ts2js must also have its last done value set to True. Otherwise bad
                        # things happen to critic values
                        if done and True not in dones:
                            dones[-1] = True
                        
                        for i, (bufstate, bufaction, bufreward, bufdone) in enumerate(zip(states, actions, rewards, dones)):
                            done  = done or bufdone
                            value = critic(bufstate).numpy()
                            buff.push(bufstate, bufaction, bufreward, value)

                            if bufdone or traj_len >= max_traj_len: # or (done and i+1 == len(dones)):
                                buff.end_trajectory(terminal_value=(not bufdone) * value)
                                break

                        recursive_sample = next_level
                    #print(traj_len, num_steps)
        return memories

class RPO(RPO_Worker):
    """
    Central RPO Worker

    Args:
        actor_stack (list): list of actor pytorch network
        critic_stack (list): list of critic pytorch network
        env_fn_stack (list): list of environment constructor function
        fixed_pol (list): list of flags for which levels in hierarchy are fixed
        a_lr (list): a_lr across hierarchy
        c_lr (list): c_lr across hierarchy
        grad_clip (list): grad_clip across hierarchy
        mirror (list): mirror loss coefficient across hierarchy
        discount (list): discount gamma across hierarchy
        paths (list): policy paths across hierarchy
        workers: number of workers
        prenormalize_steps: prenormalization steps
        redis: url to redis server for Ray object store

    Attributes:
        system: RecursiveSystem object
        discount: discount factor
        fixed_pol (list): list of flags for which levels in hierarchy are fixed
        recurrent (list): list of recurrent policy indicators across hierarhcy
        workers (list): list of ray worker IDs for sampling experience
        optims (list): list of ray worker IDs for optimizing policies
        state_mirror_fns (list): list of environment-specific functions for mirroring state for mirror loss
        action_mirror_fns (list): list of environment-specific functions for mirroring action for mirror loss
        rates (list): list of planner rates from bottom to top of hierarchy
    """

    def __init__(self, actor_stack, critic_stack, env_fn_stack, fixed_pol=[False], a_lr=[1e-4], c_lr=[1e-4], grad_clip=[0.05], mirror=[0], discount=[0.95], workers=4, prenormalize_steps=100, paths=None, redis=None, **kwargs):
        RPO_Worker.__init__(self, actor_stack, critic_stack, env_fn_stack, fixed_pol)
        self.discount = discount
        self.fixed_pol = fixed_pol

        assert len(actor_stack) == len(a_lr)
        assert len(actor_stack) == len(c_lr)
        assert len(actor_stack) == len(grad_clip)
        assert len(actor_stack) == len(mirror)
        assert len(actor_stack) == len(discount)
        assert len(actor_stack) == len(a_lr)

        if paths is None and prenormalize_steps > 1:
            while self.system.policy_stack[0].welford_state_n < prenormalize_steps:
                self.system.eval(trials=5, max_len=50, deterministic=False, update_norm=True)
            
            for a, c in zip(self.system.policy_stack, self.system.critic_stack):
                c.copy_normalizer_stats(a)

        if not ray.is_initialized():
            if redis is not None:
                ray.init(redis_address=redis)
            else:
                ray.init(num_cpus=workers + len(self.system.policy_stack))

        self.recurrent = [a.is_recurrent or c.is_recurrent for (a,c) in zip(self.system.policy_stack, self.system.critic_stack)]

        self.workers = [RPO_Sampler.remote(self.system.policy_stack, self.system.critic_stack, self.system.env_fn_stack, self.fixed_pol, discount) for _ in range(workers)]
        self.optims = []
        for a, c, lr1, lr2, clip, mirr in zip(self.system.policy_stack, 
                                                       self.system.critic_stack,
                                                       a_lr,
                                                       c_lr,
                                                       grad_clip,
                                                       mirror):
            self.optims += [PPO_Optim.remote(a, c, a_lr=lr1, c_lr=lr2, grad_clip=clip, mirror=mirr)]

        env = self.system.env_head
        self.state_mirror_fns = []
        self.action_mirror_fns = []
        for m in mirror:
            if m > 0 and hasattr(env, 'mirror_state_fnptr') and hasattr(env, 'mirror_action_fnptr'):
                self.state_mirror_fns += [env.mirror_state_fnptr()]
                self.action_mirror_fns += [env.mirror_action_fnptr()]
            else:
                self.state_mirror_fns += [None]
                self.action_mirror_fns += [None]
            if hasattr(env, 'subenv'):
                env = env.subenv
            else:
                env = None

        self.rates = []
        subenv = self.system.env_head
        while hasattr(subenv, 'planner_rate'):
            self.rates += [subenv.planner_rate]
            subenv = subenv.subenv
        self.rates = [int(x) for x in np.cumprod(self.rates[::-1])[::-1]]


    def do_iteration(self, num_steps, max_traj_len, epochs, kl_thresh=0.02, verbose=True, batch_sizes=[64], mirror=False):
        """
        Function to do a single iteration of RPO

        Args:
            max_traj_len (int): maximum trajectory length of an episode
            num_steps (int): number of steps to collect experience for
            epochs (int): optimzation epochs
            batch_size (int): optimzation batch size
            mirror (bool): Mirror loss enabled or not
            kl_thresh (float): threshold for max kl divergence
            verbose (bool): verbose logging output
            batch_sizes (list): list of batch sizes for optimization
        """
        start = time.time()
        actor_ids  = [ray.put(list(actor.parameters()))  for actor in self.system.policy_stack]
        critic_ids = [ray.put(list(critic.parameters())) for critic in self.system.critic_stack]

        steps = max(num_steps // len(self.workers), max_traj_len)

        for w in self.workers:
            w.sync_policies.remote(actor_ids, critic_ids)

        if verbose:
            print("\t{:5.2f}s to copy policy params to workers.".format(time.time() - start))

        start = time.time()
        eval_rewards = ray.get([w.evaluate.remote(trajs=1, max_traj_len=max_traj_len) for w in self.workers])
        if verbose:
            print("\t{:5.2f}s to evaluate policy stack.".format(time.time() - start))
        torch.set_num_threads(1)

        start    = time.time()
        buffers  = ray.get([w.collect_experience.remote(min_steps=steps, max_traj_len=max_traj_len) for w in self.workers])
        memory   = [merge_buffers(buffs) for buffs in zip(*buffers)]
        steps    = [len(m) for m in memory]
        trajs    = [len(m.traj_idx) for m in memory]
        traj_len = np.mean([memory[-1].traj_idx[i+1] - memory[-1].traj_idx[i] for i in range(len(memory[-1].traj_idx)-1)])

        #batch_ratio = batch_sizes[0] / steps[-1]
        #print(batch_ratio, trajs, batch_sizes)
        #input()
        #batch_sizes = [max(1, batch_sizes[0] // r) for r in self.rates] if len(batch_sizes) == 1 else batch_sizes
        batch_sizes = [batch_sizes[0] for _ in self.system.policy_stack] if len(batch_sizes) == 1 else batch_sizes

        if verbose:
            print("\t{:5.2f}s to collect and merge experience buffers.".format(time.time() - start))

        # TODO: Find way to not optimize fixed_pol policies that is more efficient and less lines of code
        start = time.time()
        rets = []
        ps = list(zip(actor_ids, critic_ids, self.optims, memory, self.recurrent, batch_sizes, epochs, self.state_mirror_fns, self.action_mirror_fns, self.fixed_pol))

        for (actor_id, critic_id, w, buff, recurrent, batch, num_epochs, state_fn, action_fn, is_fixed) in ps:
            # only optimize if not fixed
            if not is_fixed:
                w.sync_policy.remote(actor_id, critic_id)
                ray_ids = w.optimize.remote(buff,
                                            epochs=num_epochs,
                                            batch_size=batch,
                                            recurrent=recurrent,
                                            state_fn=state_fn,
                                            action_fn=action_fn)
                rets += [ray_ids]
            else:
                rets += [None]
        rets = [ray.get(r) if r is not None else r for r in rets]
        a_losses = []
        c_losses = []
        m_losses = []
        kls      = []
        for r in rets:
            if r is not None:
                a_loss, c_loss, m_loss, kl = r
                a_losses      += [a_loss]
                c_losses      += [c_loss]
                m_losses      += [m_loss]
                kls           += [kl]
            else:
                a_losses      += [None]
                c_losses      += [None]
                m_losses      += [None]
                kls           += [None]

        # TODO: prevent redundant work from policies with training disabled.
        actor_params, critic_params = [], []
        for w in self.optims:
            actor_p, critic_p = ray.get(w.retrieve_parameters.remote())
            actor_params  += [actor_p]
            critic_params += [critic_p]

        self.sync_policies(ray.put(actor_params), ray.put(critic_params))
        time.sleep(0.25)
        if verbose:
            print("\t{:5.4f}s to optimize and sync policy stack.".format(time.time() - start))

        return (np.mean(eval_rewards, axis=0),
                kls,
                a_losses,
                c_losses,
                m_losses,
                steps,
                traj_len)
    
def run_experiment(args):
    """
    Function to run a PPO experiment.

    Args:
        args: argparse namespace
    """
    from util.log import create_logger
    import locale
    locale.setlocale(locale.LC_ALL, '')

    #TODO: LOADING MULTIPLE POLICIES WITH --paths AND COMBINING THEM
    #if args.paths is not None:
    #    base_existing_args = args.paths[-1]
    #    base_args = pickle.load(open(os.path.join(base, "experiment.pkl"), "rb"))
    #    base_env = base_existing_args
    #    base_env = args.env

    env_fn_stack = []
    env_names = []
    base_envname = args.env
    base_env = env_factory(**vars(args))
    for p in args.planners:
        args.__dict__['env'] = p
        env_fn_stack += [env_factory(**vars(args))]
        env_names += [p]
    env_fn_stack += [base_env]
    env_head = env_fn_stack[0](env_fn_stack[1:])
    args.__dict__['env'] = base_envname

    d = len(env_fn_stack)-1

    assert len(args.layers)     == 1 or len(args.layers)     == d
    assert len(args.arch)       == 1 or len(args.arch)       == d
    assert len(args.std)        == 1 or len(args.std)        == d
    assert len(args.a_lr)       == 1 or len(args.a_lr)       == d
    assert len(args.c_lr)       == 1 or len(args.c_lr)       == d
    assert len(args.grad_clip)  == 1 or len(args.grad_clip)  == d
    assert len(args.batch_size) == 1 or len(args.batch_size) == d
    assert len(args.discount)   == 1 or len(args.discount)   == d
    assert len(args.kl)         == 1 or len(args.kl)         == d
    assert len(args.epochs)     == 1 or len(args.epochs)     == d
    assert len(args.mirror)     == 1 or len(args.mirror)     == d
    assert args.paths           == None or len(args.paths)   == d
    assert len(args.fixed_pol)  == 1 or len(args.fixed_pol)  == d
    assert len(args.bounded)    == 1 or len(args.bounded)    == d

    if len(args.layers) == 1:
        args.__dict__['layers'] = [[int(x) for x in args.layers[0].split(',')] for _ in range(d)]
    else:
        args.__dict__['layers'] = [[int(x) for x in l.split(',')] for l in args.layers]
    args.__dict__['arch']       = [args.arch[0]       for _ in range(d)] if len(args.arch)       == 1 else args.arch
    args.__dict__['std']        = [args.std[0]        for _ in range(d)] if len(args.std)        == 1 else args.std
    args.__dict__['a_lr']       = [args.a_lr[0]       for _ in range(d)] if len(args.a_lr)       == 1 else args.a_lr
    args.__dict__['c_lr']       = [args.c_lr[0]       for _ in range(d)] if len(args.c_lr)       == 1 else args.c_lr
    args.__dict__['grad_clip']  = [args.grad_clip[0]  for _ in range(d)] if len(args.grad_clip)  == 1 else args.grad_clip
    args.__dict__['discount']   = [args.discount[0]   for _ in range(d)] if len(args.discount)   == 1 else args.discount
    args.__dict__['kl']         = [args.kl[0]         for _ in range(d)] if len(args.kl)         == 1 else args.kl
    args.__dict__['epochs']     = [args.epochs[0]     for _ in range(d)] if len(args.epochs)     == 1 else args.epochs
    args.__dict__['mirror']     = [args.mirror[0]     for _ in range(d)] if len(args.mirror)     == 1 else args.mirror
    args.__dict__['fixed_pol']  = [args.fixed_pol[0]  for _ in range(d)] if len(args.fixed_pol)  == 1 else args.fixed_pol
    args.__dict__['bounded']    = [args.bounded[0]    for _ in range(d)] if len(args.bounded)    == 1 else args.bounded

    # convert 0/1 into False/True
    args.__dict__['fixed_pol'] = [True if a != 0 else False for a in args.__dict__['fixed_pol']]
    args.__dict__['bounded'] = [True if a != 0 else False for a in args.__dict__['bounded']]

    paths  = args.paths if args.paths is not None else ['' for _ in range(d)]
    paths = ['' if p == 'None' else p for p in paths]

    e = env_head
    obs_dims = []
    act_dims = []
    while True:
        obs_dims += [e.observation_space.shape[0]]
        act_dims += [e.action_space.shape[0]]

        if hasattr(e, 'subenv') and not e.has_pd_env:
            e = e.subenv
        else:
            break

    actor_stack  = []
    critic_stack = []
    for name, path, i, o, arch, ls, std, bounded in zip(env_names, paths, obs_dims, act_dims, args.arch, args.layers, args.std, args.bounded):
        if path != '':
            # get experiment.pkl file
            run_args = pickle.load(open(os.path.join(path, "experiment.pkl"), "rb"))
            # print(run_args.__dict__)
            assert(run_args.__dict__['impedance'] == args.impedance)
            assert(run_args.__dict__['height'] == args.height)
            assert(run_args.__dict__['task'] == args.task)
            assert(run_args.__dict__['perception'] == args.perception)
            assert(run_args.__dict__['gaze_control'] == args.gaze_control)
            # assert(run_args.__dict__['auto_clock'] == args.auto_clock)

            suffix = name + '-' + base_envname
            actor_stack.append(torch.load(os.path.join(path, f"actor_{suffix}.pt")))
            critic_stack.append(torch.load(os.path.join(path, f"critic_{suffix}.pt")))
            print(f"Loaded actor/critic files for {name} (level {len(actor_stack)}) in env stack.")

        else:
            if arch.lower() == 'ff':
                actor_stack.append(FF_Stochastic_Actor(i, o,
                                                        bounded=bounded,
                                                        fixed_std=torch.ones(o)*std,
                                                        layers=ls))
                critic_stack.append(FF_V(i, layers=ls))
            elif arch.lower() == 'lstm':
                actor_stack.append(LSTM_Stochastic_Actor(i, o,
                                                        bounded=bounded,
                                                        fixed_std=torch.ones(o)*std,
                                                        layers=ls))
                critic_stack.append(LSTM_V(i, layers=ls))
            elif arch.lower() == 'gru':
                actor_stack.append(GRU_Stochastic_Actor(i, o,
                                                        bounded=bounded,
                                                        fixed_std=torch.ones(o)*std,
                                                        layers=ls))
                critic_stack.append(GRU_V(i, layers=ls))
            else:
                raise RuntimeError

    if args.wandb:
        import wandb
        wandb.init(group = args.run_name, project="cassie-hrl", config=args, sync_tensorboard=True)

    algo = RPO(actor_stack, critic_stack, env_fn_stack, **vars(args))

    # create a tensorboard logging object
    if not args.nolog:
        logger = create_logger(args)
        
        for i, name in enumerate(env_names):
            suffix = name + '-' + base_envname
            p = os.path.join(logger.dir, f'actor_{suffix}.pt')
            torch.save(algo.system.policy_stack[i], p)
            p = os.path.join(logger.dir, f'critic_{suffix}.pt')
            torch.save(algo.system.critic_stack[i], p)

    else:
        logger = None

    itr = 0
    timesteps = [0 for _ in actor_stack]
    best_reward = [None for _ in actor_stack]
    fixed_pol = args.fixed_pol
    while timesteps[-1] < args.timesteps:
        eval_rewards, kls, a_losses, c_losses, m_losses, steps, traj_len = algo.do_iteration(args.num_steps,
                                                                                             args.traj_len,
                                                                                             args.epochs,
                                                                                             batch_sizes=args.batch_size,
                                                                                             kl_thresh=args.kl,
                                                                                             mirror=args.mirror)

        timesteps = [t + dt for (t,dt) in zip(timesteps, steps)]
        print("iter {:4d}, mean eplen {:5.1f}, timesteps {}".format(itr, traj_len, ''.join(['{:n} | '.format(t) for t in timesteps])))
        for i, (name, r, kl, a_loss, c_loss, m_loss, timestep) in enumerate(zip(env_names, eval_rewards, kls, a_losses, c_losses, m_losses, timesteps)):
            suffix = name + '-' + base_envname
            if not fixed_pol[i]:
                print("\t{} (level {:2d}): return {:5.2f} | KL {:5.4f} | Actor loss {:5.4f} | Critic Loss {:5.4f}".format(name, i+1, r, kl, a_loss, c_loss), end='')
                if m_loss != 0:
                    print(" | Mirror Loss {:6.5f} | ".format(m_loss), end='')
                print()
                if best_reward[i] is None or r > best_reward[i]:
                    best_reward[i] = r
                    if not args.nolog:
                        p = os.path.join(logger.dir, f'actor_{suffix}.pt')
                        torch.save(algo.system.policy_stack[i], p)
                        print("\t\t(best policy so far! saving to {})".format(p))

                    if not args.nolog:
                        p = os.path.join(logger.dir, f'critic_{suffix}.pt')
                        torch.save(algo.system.critic_stack[i], p)

                if logger is not None:
                    logger.add_scalar(args.env + '/kl-'+suffix, kl, timestep)
                    logger.add_scalar(args.env + '/return-'+suffix, r, timestep)
                    logger.add_scalar(args.env + '/actor loss-'+suffix, a_loss, timestep)
                    logger.add_scalar(args.env + '/critic loss-'+suffix, c_loss, timestep)
                    if m_loss != 0:
                        logger.add_scalar(args.env + '/mirror loss-'+suffix, m_loss, timestep)
                    #logger.add_scalar(args.env + '/sample rate'+str(i+1), times[0], timesteps)
                    #logger.add_scalar(args.env + '/update time'+str(i+1), times[1], timesteps)
        if logger is not None:
            logger.add_scalar(args.env + '/eplen', traj_len, timestep)
            itr += 1
    print("Finished ({} of {}).".format(timesteps[-1], args.timesteps))

    if args.wandb:
        wandb.join()
