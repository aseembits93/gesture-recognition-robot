import numpy as np
import os, time

import matplotlib.pyplot as plt

SAMPLES = 1000

class Live_Gait_Plot:
    def __init__(self, behavior):
        self.behavior = behavior

    def __call__(self, pipe):
        print('starting live gait plotter...')

        self.pipe = pipe
        self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5))
        self.axs[0].set_title('Live PeriodicFcn Plot')
        self.axs[0].set_ylabel("𝔼[l_force]")
        self.axs[1].set_ylabel("𝔼[r_force]")

        self.xlim = sum(self.behavior.ratio)
        self.x = np.linspace(0, self.xlim, num=SAMPLES)

        self.axs[0].set_xlim(left=0.0, right=self.xlim)
        self.axs[1].set_xlim(left=0.0, right=self.xlim)
        self.axs[0].set_ylim(bottom=-1.1, top=0.1)
        self.axs[1].set_ylim(bottom=-1.1, top=0.1)

        self.fig.canvas.draw()  # matplotlib is stupid, becuase this needs to happen before draw_artist()

        # cache background
        self.ax0background = self.fig.canvas.copy_from_bbox(self.axs[0].bbox)
        self.ax1background = self.fig.canvas.copy_from_bbox(self.axs[1].bbox)

        left_foot = self.behavior._vonmises_func(self.x, self.behavior.phase_dict['left force'])
        right_foot = self.behavior._vonmises_func(self.x, self.behavior.phase_dict['right force'])

        self.l0, = self.axs[0].plot(self.x, left_foot, label='𝔼[l_force]')
        self.l1, = self.axs[1].plot(self.x, right_foot, label='𝔼[r_force]')

        timer = self.fig.canvas.new_timer(interval=5)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                behavior = command
                self.xlim = sum(behavior.ratio)
                self.x = np.linspace(0, self.xlim)

                self.axs[0].set_xlim(left=0.0, right=self.xlim)
                self.axs[1].set_xlim(left=0.0, right=self.xlim)

                left_foot = behavior._vonmises_func(self.x, behavior.phase_dict['left force'])
                right_foot = behavior._vonmises_func(self.x, behavior.phase_dict['right force'])

                self.l0.set_data(self.x, left_foot)
                self.l1.set_data(self.x, right_foot)

                # restore background
                self.fig.canvas.restore_region(self.ax0background)
                self.fig.canvas.restore_region(self.ax1background)

                # redraw just the points
                self.axs[0].draw_artist(self.l0)
                self.axs[1].draw_artist(self.l1)

                # fill in axes rectangle
                self.fig.canvas.blit(self.axs[0].bbox)
                self.fig.canvas.blit(self.axs[1].bbox)

                self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        return True

class Gait_Plot:
    def __init__(self, behavior):
        plt.ion()
        self.fig, self.axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,5))

        self.xlim = sum(behavior.ratio)
        self.x = np.linspace(0, self.xlim)

        self.axs[0].set_xlim(left=0.0, right=self.xlim)
        self.axs[1].set_xlim(left=0.0, right=self.xlim)
        self.axs[0].set_ylim(bottom=-1.1, top=0.1)
        self.axs[1].set_ylim(bottom=-1.1, top=0.1)

        self.fig.canvas.draw()  # matplotlib is stupid, becuase this needs to happen before draw_artist

        # cache background
        self.ax0background = self.fig.canvas.copy_from_bbox(self.axs[0].bbox)
        self.ax1background = self.fig.canvas.copy_from_bbox(self.axs[1].bbox)

        left_foot = behavior._vonmises_func(self.x, behavior.phase_dict['left force'])
        right_foot = behavior._vonmises_func(self.x, behavior.phase_dict['left force'])

        self.l0, = self.axs[0].plot(self.x, left_foot, label='𝔼[l_force]')
        self.l1, = self.axs[1].plot(self.x, right_foot, label='𝔼[r_force]')

        plt.legend()
        time.sleep(0.5)

    def update(self, behavior):

        self.xlim = sum(behavior.ratio)
        self.x = np.linspace(0, self.xlim)

        self.axs[0].set_xlim(left=0.0, right=self.xlim)
        self.axs[1].set_xlim(left=0.0, right=self.xlim)

        left_foot = behavior._vonmises_func(self.x, behavior.phase_dict['left force'])
        right_foot = behavior._vonmises_func(self.x, behavior.phase_dict['left force'])

        self.l0.set_data(self.x, left_foot)
        self.l1.set_data(self.x, right_foot)

        # restore background
        self.fig.canvas.restore_region(self.ax0background)
        self.fig.canvas.restore_region(self.ax1background)

        # redraw just the points
        self.axs[0].draw_artist(self.l0)
        self.axs[1].draw_artist(self.l1)

        # fill in axes rectangle
        self.fig.canvas.blit(self.axs[0].bbox)
        self.fig.canvas.blit(self.axs[1].bbox)

        self.fig.canvas.flush_events()
