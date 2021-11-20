import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time, os, sys

from util.env import env_factory

import argparse
import pickle

# Get policy to plot from args, load policy and env
parser = argparse.ArgumentParser()
# General args
parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
parser.add_argument("--pre_steps", type=int, default=300, help="Number of \"presteps\" to take for the policy to stabilize before data is recorded")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to record")
parser.add_argument("--pre_speed", type=float, default=1.0, help="Commanded action during the presteps")
parser.add_argument("--plot_speed", type=float, default=1.0, help="Commanded action during the actual recorded steps")

parser.add_argument("--shift", type=float, default=0.5, help="Shift parameter for right foot")
parser.add_argument("--ratio", type=float, default=0.6, help="Ratio of time in stance phase")
parser.add_argument("--logfilename", type=str, default="data", help="Name of data logfile for this plot test")
args = parser.parse_args()

run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

print("pre steps: {}\tnum_steps: {}".format(args.pre_steps, args.num_steps))
print("pre speed: {}\tplot_speed: {}".format(args.pre_speed, args.plot_speed))

# Load environment
# env_fn = env = env = env_factory(run_args.env, simrate=run_args.simrate, dynamics_randomization=run_args.dyn_random, reward=run_args.reward, history=run_args.history)
env_fn = env_factory(**vars(run_args))
cassie_env = env_fn()
cassie_env.dynamics_randomization = False
cassie_env.evaluation_mode = True
cassie_env.stairs = False

def reset_env():
    # env initial set up
    state = cassie_env.reset()
    cassie_env.set_behavior([args.ratio, 1-args.ratio], [0.0, args.shift])
    cassie_env.speed = args.pre_speed
    cassie_env.side_speed = 0.0
    return state
state = reset_env()

# Load policy
policy = torch.load(os.path.join(args.path, "actor.pt"))
policy.eval()
if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

obs_dim = cassie_env.observation_space.shape[0]  # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

no_delta = True
impedance = True
offset = cassie_env.offset

num_steps = args.num_steps
pre_steps = args.pre_steps
simrate = cassie_env.simrate
torques = np.zeros((num_steps*simrate, 10))
GRFs = np.zeros((num_steps*simrate, 2))
targets = np.zeros((num_steps*simrate, 10))
P_gains = np.zeros((num_steps*simrate, 10))
D_gains = np.zeros((num_steps*simrate, 10))
heights = np.zeros(num_steps*simrate)
speeds = np.zeros(num_steps*simrate)
foot_pos = np.zeros((num_steps*simrate, 6))
mj_foot_pos = np.zeros((num_steps*simrate, 6))
foot_vel = np.zeros((num_steps*simrate, 6))
if impedance:
    actions = np.zeros((num_steps*simrate, 30))
else:
    actions = np.zeros((num_steps*simrate, 10))
pelaccel = np.zeros(num_steps*simrate)
pelheight = np.zeros(num_steps*simrate)
act_diff = np.zeros(num_steps*simrate)
actuated_pos = np.zeros((num_steps*simrate, 10))
actuated_vel = np.zeros((num_steps*simrate, 10))
torque_cost = np.zeros(num_steps*simrate)
motor_pos = np.zeros((num_steps*simrate, 10))
prev_action = None
pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

# Execute policy and save torques
with torch.no_grad():
    failed = False
    while not failed:
        state = reset_env()
        qpos = cassie_env.sim.qpos()
        qpos[2] += 0.4
        cassie_env.sim.set_qpos(qpos)
        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()
        i = 0
        while i < pre_steps:
            i += 1
            action = policy.forward(torch.Tensor(state), deterministic=True).numpy()
            state, reward, done, _ = cassie_env.step(action)
            if done:
                # set failed
                failed = True
                # retry
                break
            state = torch.Tensor(state)
            cassie_env.render()
        if not failed:
            break
    cassie_env.speed = args.plot_speed
    # while cassie_env.phase != 0:
    #     print(cassie_env.phase)
    #     action = policy.forward(torch.Tensor(state), deterministic=True).numpy()
    #     state, reward, done, _ = cassie_env.step(action)
    #     state = torch.Tensor(state)
    #     cassie_env.render()
    for i in range(num_steps):

        # reset variables normally reset in step()
        cassie_env.sim_height   = 0

        # get action
        action = policy.forward(torch.Tensor(state), deterministic=True).numpy()

        for j in range(simrate):
            if no_delta:
                if impedance:
                    target = action[:10] + offset
                    P_gains[i*simrate+j, :] = action[10:20]
                    D_gains[i*simrate+j, :] = action[20:]
                else:
                    target = action + offset
            real_action = action
            actions[i*simrate+j, :] = action
            targets[i*simrate+j, :] = target
            # print(target)

            zero2zero_clock = 0.5*(np.cos(2*np.pi/(cassie_env.phase_len+1)*(cassie_env.phase-(cassie_env.phase_len+1)/2)) + 1)
            one2one_clock = 0.5*(np.cos(2*np.pi/(cassie_env.phase_len+1)*cassie_env.phase) + 1)

            cassie_env.step_simulation(real_action)
            curr_qpos = cassie_env.sim.qpos()
            curr_qvel = cassie_env.sim.qvel()
            motor_pos[i*simrate+j, :] = np.array(curr_qpos)[pos_idx]
            torques[i*simrate+j, :] = cassie_env.cassie_state.motor.torque[:]
            torque_cost[i*simrate+j] = zero2zero_clock*0.006*np.linalg.norm(np.square(torques[i*simrate+j, [0, 1]]))
            GRFs[i*simrate+j, :] = cassie_env.sim.get_foot_forces()
            heights[i*simrate+j] = curr_qpos[2]
            speeds[i*simrate+j] = cassie_env.sim.qvel()[0]
            curr_foot = np.concatenate((cassie_env.cassie_state.leftFoot.position, cassie_env.cassie_state.rightFoot.position))
            curr_foot += np.concatenate((cassie_env.cassie_state.pelvis.position, cassie_env.cassie_state.pelvis.position))
            mj_foot = np.array(cassie_env.sim.foot_pos())
            mj_foot_pos[i*simrate+j, :] = mj_foot
            foot_pos[i*simrate+j, :] = curr_foot
            # print("left foot height: ", cassie_env.cassie_state.leftFoot.position[2])
            foot_vel[i*simrate+j, :] = np.concatenate((cassie_env.cassie_state.leftFoot.footTranslationalVelocity, cassie_env.cassie_state.rightFoot.footTranslationalVelocity))
            pelaccel[i*simrate+j] = cassie_env.cassie_state.pelvis.translationalAcceleration[2]#np.linalg.norm(cassie_env.cassie_state.pelvis.translationalAcceleration)
            pelheight[i*simrate+j] = cassie_env.cassie_state.pelvis.position[2]
            actuated_pos[i*simrate+j, :] = [curr_qpos[k] for k in pos_idx]
            actuated_vel[i*simrate+j, :] = [curr_qvel[k] for k in vel_idx]
            if prev_action is not None:
                act_diff[i*simrate+j] = np.linalg.norm(action - prev_action)
            else:
                act_diff[i*simrate+j] = 0
        prev_action = action

        cassie_env.time  += 1
        cassie_env.phase += cassie_env.phase_add

        if cassie_env.phase > cassie_env.phase_len:
            cassie_env.phase = 0

        state = cassie_env.get_full_state()
        cassie_env.render()

# Graph torque data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Torque")
ax[1][0].set_ylabel("Torque")
for i in range(5):
    ax[0][i].plot(t, torques[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, torques[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.025 sec)")

plt.tight_layout()
plt.savefig(os.path.join(args.path, "torques.png"))

# Save GRF data
# path_to_save = os.path.join(args.path, f"{args.logfilename}.pkl")
path_to_save = f"{args.logfilename}.pkl"
with open(path_to_save, "wb") as f:
    pickle.dump(
        {
            "left": GRFs[:, 0],
            "right": GRFs[:, 1],
        }, f)

# Graph GRF data
fig, ax = plt.subplots(2, figsize=(10, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
ax[0].set_ylabel("GRFs")

ax[0].plot(t, GRFs[:, 0])
ax[0].set_title("Left Foot")
ax[0].set_xlabel("Timesteps (0.025 sec)")
ax[1].plot(t, GRFs[:, 1])
ax[1].set_title("Right Foot")
ax[1].set_xlabel("Timesteps (0.025 sec)")

plt.tight_layout()
plt.savefig(os.path.join(args.path, "GRFs.png"))

# Graph PD target data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("PD Target")
ax[1][0].set_ylabel("PD Target")
for i in range(5):
    ax[0][i].plot(t, targets[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, targets[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.025 sec)")

plt.tight_layout()
plt.savefig(os.path.join(args.path, "targets.png"))

# Graph action data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Action")
ax[1][0].set_ylabel("Action")
for i in range(5):
    ax[0][i].plot(t, actions[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, actions[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.025 sec)")

plt.tight_layout()
plt.savefig(os.path.join(args.path, "actions.png"))

# If gain policy graph P gains, setpoints, torques
if impedance:
    fig, ax = plt.subplots(5, 2, sharey=True, figsize=(15, 15))
    t = np.linspace(0, num_steps-1, num_steps*simrate)
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    for i in range(5):
        ax[i][0].set_ylabel("% Change in P Gain")
        ax[i][0].plot(t, P_gains[:, i] + cassie_env.P[i])
        ax[i][0].set_title("Left " + titles[i])
        ax[i][1].plot(t, P_gains[:, i+5] + cassie_env.P[i])
        ax[i][1].set_title("Right " + titles[i])
        ax[i][1].set_xlabel("Timesteps (0.025 sec)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.path, "p_gains.png"))

    fig, ax = plt.subplots(6, 2, figsize=(15, 15), sharey=False, sharex=False)
    t = np.linspace(0, num_steps-1, num_steps*simrate)
    ax[0][0].plot(t, GRFs[:, 0], label='GRFs')
    ax[0][1].plot(t, GRFs[:, 1], label='GRFs')
    ax[0][0].set_title("Left GRFs")
    ax[0][1].set_title("Right GRFs")
    titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
    for i in range(5):
        ax[i+1][0].plot(t, 100*P_gains[:, i] / cassie_env.P[i], label='% change in P')
        ax[i+1][0].plot(t, actions[:, i], label='setpoint')
        ax[i+1][0].plot(t, torques[:, i]/100, label='torque/100')
        ax[i+1][0].set_title("Left " + titles[i])
        ax[i+1][0].set_xlabel("Timesteps (0.025 sec)")
        ax[i+1][1].plot(t, 100*P_gains[:, i+5] / cassie_env.P[i], label='% change in P')
        ax[i+1][1].plot(t, actions[:, i+5], label='setpoint')
        ax[i+1][1].plot(t, torques[:, i+5]/100, label='torque/100')
        ax[i+1][1].set_title("Right " + titles[i])
        ax[i+1][1].set_xlabel("Timesteps (0.025 sec)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(args.path, "gains_setpoints_torques.png"))

# Graph motor pos data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Motor Angle")
ax[1][0].set_ylabel("Motor Angle")
for i in range(5):
    ax[0][i].plot(t, motor_pos[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, motor_pos[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.025 sec)")

plt.tight_layout()
plt.savefig(os.path.join(args.path, "motor_pos.png"))

# Graph state data
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
ax[0][0].set_ylabel("norm")
ax[0][0].plot(t, pelaccel[:])
ax[0][0].set_title("Pel Z Accel")
ax[0][1].set_ylabel("m/s")
ax[0][1].plot(t, np.linalg.norm(torques, axis=1))
ax[0][1].set_title("Torque Norm")
titles = ["Left", "Right"]
for i in range(2):
    ax[1][i].plot(t, mj_foot_pos[:, 3*i+2])
    ax[1][i].set_title(titles[i] + " Foot")
    ax[1][i].set_xlabel("Timesteps (0.025 sec)")

plt.tight_layout()
plt.savefig(os.path.join(args.path, "state.png"))

# Graph feet qpos data
fig, ax = plt.subplots(5, 2, figsize=(12, 6), sharex=True, sharey='row')
t = np.linspace(0, num_steps*simrate*0.0005, num_steps*simrate)
ax[3][0].set_xlabel("Time (sec)")
ax[3][1].set_xlabel("Time (sec)")
sides = ["Left", "Right"]
titles = [" Foot Z Position", " Foot X Velocity", " Foot Y Velocity", " Foot Z Velocity"]
for i in range(2):
    # ax[0][i].plot(t, foot)
    ax[0][i].plot(t, foot_pos[:, 3*i+2])
    ax[0][i].set_title(sides[i] + titles[0])
    ax[0][i].set_ylabel("Z Position (m)")
    ax[1][i].plot(t, mj_foot_pos[:, 3*i+2])
    ax[1][i].set_title(sides[i] + " mj foot z pos")
    ax[1][i].set_ylabel("Z Position (m)")
    for j in range(3):
        ax[j+2][i].plot(t, foot_vel[:, 3*i+j])
        ax[j+2][i].set_title(sides[i] + titles[j+1])
        ax[j+2][i].set_ylabel("Velocity (m/s)")    

plt.tight_layout()
plt.savefig(os.path.join(args.path, "feet.png"))

# Graph phase portrait for actuated joints
fig, ax = plt.subplots(1, 5, figsize=(15, 4))
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0].set_ylabel("Velocity")
# ax[1][0].set_ylabel("Velocity")
for i in range(5):
    ax[i].plot(actuated_pos[:, i], actuated_vel[:, i])
    ax[i].plot(actuated_pos[:, i+5], actuated_vel[:, i+5])
    ax[i].set_title(titles[i])
    # ax[1][i].plot(actuated_pos[:, i+5], actuated_vel[:, i+5])
    # ax[1][i].set_title("Right " + titles[i])
    ax[i].set_xlabel("Angle")

plt.tight_layout()
plt.savefig(os.path.join(args.path, "phaseportrait.png"))

# Misc Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
t = np.linspace(0, num_steps-1, num_steps*simrate)
# ax.set_ylabel("norm")
# ax.set_title("Action - Prev Action Norm")
# ax.plot(t, act_diff)
# ax.set_ylabel("Height (m)")
# ax.set_title("Torque Cost")
ax.plot(t, torque_cost)
# sides = ["Left", "Right"]
# for i in range(2):
#     ax[0].plot(t, actuated_vel[:, 5*i], label=sides[i] + " Roll Vel")
#     ax[0].plot(t, actuated_vel[:, 1+5*i], label=sides[i] + " Yaw Vel")
#     ax[1].plot(t, torques[:, 5*i], label=sides[i] + " Roll Torque")
#     ax[1].plot(t, torques[:, 1+5*i], label=sides[i] + " Yaw Torque")

# ax[0].legend()
# ax[1].legend()
plt.savefig(os.path.join(args.path, "misc.png"))
