import torch
import numpy as np
import locale
import os
import time
import sys

from util.env import env_factory
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_hiddens(policy):
    hiddens = []
    if hasattr(policy, 'hidden'):
        hiddens += [h.data for h in policy.hidden]

    if hasattr(policy, 'cells'):
        hiddens += [c.data for c in policy.cells]

    return torch.cat([layer.view(-1) for layer in hiddens]).numpy()

class Live_PCA_Plot:
    def __init__(self, policy, env):
        env.dynamics_randomization = False
        basis = []
        print("Gathering PCA data...")
        eps = 2
        for episode in range(eps):
            state = env.reset()

            done = False
            timesteps = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            print("\tcollecting episode {:2d} of {:2d}".format(episode+1, eps))
            while not done and timesteps < 1000:

                action = policy.forward(torch.Tensor(state)).detach().numpy()
                state, reward, done, _ = env.step(action)
                timesteps += 1

                memory = get_hiddens(policy)
                basis.append(memory)
        basis = np.vstack(basis)

        self.pca = PCA(n_components=2)
        self.pca.fit(basis)
        # plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.pca_length = 10

        self.points = []
        point = get_hiddens(policy)
        self.points.append(point)

        c = []
        for i in range(self.pca_length):
            c.append(
                np.hstack([(0, 0.05, 0.05), (len(self.points) - i/2) / len(self.points)]))
        self.c = c

        tmp = self.pca.transform(np.vstack(self.points))
        x = tmp[:, 0]
        y = tmp[:, 1]

        plt.title('Dynamic Plot of LSTM Memory')
        plt.xlabel('Component 0', fontsize=18)
        plt.ylabel('Component 1', fontsize=18)
        self.line, = self.ax.plot(x, y, color=c[len(self.points)])
        print("Finished creating PCA plot.")

    def __call__(self, pipe):
        self.pipe = pipe
        timer = self.figure.canvas.new_timer(interval=5)
        timer.add_callback(self.update)
        timer.start()

        print('...done')
        plt.show()

    def terminate(self):
        plt.close('all')

    def update(self):

        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                point = command
                self.points.append(point)

                if len(self.points) > self.pca_length:
                    self.points = self.points[1:]

                tmp = self.pca.transform(np.vstack(self.points))
                x = tmp[:, 0]
                y = tmp[:, 1]

                plt.ylim(bottom=-10, top=10)
                plt.xlim(left=-10, right=10)
                plt.axis('off')

                time.sleep(0.01)
                self.line.set_xdata(x)
                self.line.set_ydata(y)
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
        return True


class PCA_Plot:
    def __init__(self, policy, env):
        env.dynamics_randomization = False
        basis = []
        print("Gathering PCA data...")
        eps = 2
        for episode in range(eps):
            state = env.reset()

            done = False
            timesteps = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            print("\tcollecting episode {:2d} of {:2d}".format(episode+1, eps))
            while not done and timesteps < 1000:

                action = policy.forward(torch.Tensor(state)).detach().numpy()
                state, reward, done, _ = env.step(action)
                timesteps += 1

                memory = get_hiddens(policy)
                basis.append(memory)
        basis = np.vstack(basis)

        self.pca = PCA(n_components=2)
        self.pca.fit(basis)
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.pca_length = 10

        self.points = []
        point = get_hiddens(policy)
        self.points.append(point)

        c = []
        for i in range(self.pca_length):
            c.append(
                np.hstack([(0, 0.05, 0.05), (len(self.points) - i/2) / len(self.points)]))
        self.c = c

        tmp = self.pca.transform(np.vstack(self.points))
        x = tmp[:, 0]
        y = tmp[:, 1]

        plt.title('Dynamic Plot of LSTM Memory')
        plt.xlabel('Component 0', fontsize=18)
        plt.ylabel('Component 1', fontsize=18)
        self.line, = self.ax.plot(x, y, color=c[len(self.points)])
        time.sleep(0.5)
        print("Finished creating PCA plot.")

    def update(self, policy):
        point = get_hiddens(policy)
        self.points.append(point)

        if len(self.points) > self.pca_length:
            self.points = self.points[1:]

        tmp = self.pca.transform(np.vstack(self.points))
        x = tmp[:, 0]
        y = tmp[:, 1]

        plt.ylim(bottom=-10, top=10)
        plt.xlim(left=-10, right=10)
        plt.axis('off')

        self.line.set_xdata(x)
        self.line.set_ydata(y)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Need file.")
        raise RuntimeError

    max_traj_len = 400

    policy = torch.load(sys.argv[1])

    if len(sys.argv) < 3:
        reward = 'r2l_clock'
    else:
        reward = sys.argv[2]

    env = env_factory(policy.env_name, reward=reward)()

    pca_obj = PCA_Plot(policy, env)
    env.dynamics_randomization = False
    state = env.reset()

    while True:
        done = False
        points = None
        timesteps = 0
        state = env.reset()

        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()

        while not done and timesteps < max_traj_len:
            timesteps += 1

            env.speed = 0.5
            env.phase_add = int(env.simrate)
            env.side_speed = 0.0

            action = policy.forward(state).detach().numpy()
            state, _, done, _ = env.step(action)
            env.render()
            time.sleep(0.01)
            pca_obj.update(policy)
