import numpy as np
import locale
import os
import time
import sys

from util.env import env_factory
import matplotlib.pyplot as plt

class Live_PD_Plot:
    def __init__(self):
        pass

    def __call__(self, pipe):
        print('starting live pds plotter...')
        self.pipe = pipe

        self.figure, self.axs = plt.subplots(ncols=2, nrows=6, figsize=(8, 15))

        xlim = 120

        self.left_grfs = np.zeros(xlim)
        self.right_grfs = np.zeros(xlim)

        self.figure.canvas.draw()  # matplotlib is stupid, becuase this needs to happen before draw_artist()

        # cache background
        self.bgs = [[self.figure.canvas.copy_from_bbox(self.axs[i][j].bbox) for j in range(2)] for i in range(6)]

        self.data = {}
        xs = range(120)
        for i, s in enumerate(['left', 'right']):
            self.data[s] = {}
            for j, l in enumerate(['hip roll', 'hip yaw', 'hip pitch', 'knee', 'foot']):
                self.data[s][l] = {}
                self.data[s][l]['P'] = np.zeros(xlim)
                self.data[s][l]['setpoint'] = np.zeros(xlim)
                self.data[s][l]['line1'], = self.axs[1 + j][i].plot(xs, self.data[s][l]['P'], label='P')
                self.data[s][l]['line2'], = self.axs[1 + j][i].plot(xs, self.data[s][l]['setpoint'], label='setpoint')

        self.grfline1, = self.axs[0][0].plot(xs, self.left_grfs)
        self.grfline2, = self.axs[0][1].plot(xs, self.right_grfs)
        plt.legend()

        timer = self.figure.canvas.new_timer(interval=10)
        timer.add_callback(self.update)
        timer.start()
        
        print('...done')
        plt.show()

    def update(self):

        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                action, left_grfs, right_grfs = command

                self.left_grfs = np.hstack((self.left_grfs[1:], left_grfs))
                self.right_grfs = np.hstack((self.right_grfs[1:], right_grfs))

                self.axs[0][0].set_ylim(bottom=0, top=np.max(self.left_grfs))
                self.axs[0][0].set_xlim(left=0, right=120)
                self.axs[0][1].set_ylim(bottom=0, top=np.max(self.right_grfs))
                self.axs[0][1].set_xlim(left=0, right=120)

                self.grfline1.set_ydata(self.left_grfs)
                self.grfline2.set_ydata(self.right_grfs)

                for i, s in enumerate(['left', 'right']):
                    for j, l in enumerate(['hip roll', 'hip yaw', 'hip pitch', 'knee', 'foot']):
                        setpoint = action[i * 5 + j]
                        pgain = action[10 + i * 5 + j]
                        d1 = np.hstack((self.data[s][l]['P'][1:], pgain))
                        d2 = np.hstack((self.data[s][l]['setpoint'][1:], setpoint))
                        self.data[s][l]['line1'].set_ydata(d1)
                        self.data[s][l]['line2'].set_ydata(d2)
                        self.data[s][l]['P'] = d1
                        self.data[s][l]['setpoint'] = d2
                        self.axs[1+j][i].set_ylim(bottom=-1, top=1)

                # self.figure.canvas.draw()  # not using blit
                # Draw GRFs
                self.figure.canvas.restore_region(self.bgs[0][0])
                self.figure.canvas.restore_region(self.bgs[0][1])
                self.axs[i][0].draw_artist(self.grfline1)
                self.axs[i][1].draw_artist(self.grfline2)
                self.figure.canvas.blit(self.axs[0][0].bbox)
                self.figure.canvas.blit(self.axs[0][1].bbox)

                # Draw rest
                for i, s in enumerate(['left', 'right']):
                    for j, l in enumerate(['hip roll', 'hip yaw', 'hip pitch', 'knee', 'foot']):
                        # restore background
                        self.figure.canvas.restore_region(self.bgs[1+j][i])
                        # redraw just the points
                        self.axs[1+j][i].draw_artist(self.data[s][l]['line1'])
                        self.axs[1+j][i].draw_artist(self.data[s][l]['line2'])
                        # fill in axes rectangle
                        self.figure.canvas.blit(self.axs[1+j][i].bbox)
                        self.figure.canvas.blit(self.axs[1+j][i].bbox)

                self.figure.canvas.flush_events()
        return True

class PD_Plot:
    def __init__(self, policy, env):
        plt.ion()
        self.figure, self.axs = plt.subplots(ncols=2, nrows=6, figsize=(8, 15))

        xlim = 120

        self.left_grfs = np.zeros(xlim)
        self.right_grfs = np.zeros(xlim)

        self.data = {}
        xs = range(120)
        for i, s in enumerate(['left', 'right']):
            self.data[s] = {}
            for j, l in enumerate(['hip roll', 'hip yaw', 'hip pitch', 'knee', 'foot']):
                self.data[s][l] = {}
                self.data[s][l]['P'] = np.zeros(xlim)
                self.data[s][l]['setpoint'] = np.zeros(xlim)
                self.data[s][l]['line1'], = self.axs[1 +
                                                     j][i].plot(xs, self.data[s][l]['P'], label='P')
                self.data[s][l]['line2'], = self.axs[1 +
                                                     j][i].plot(xs, self.data[s][l]['setpoint'], label='setpoint')

        self.grfline1, = self.axs[0][0].plot(xs, self.left_grfs)
        self.grfline2, = self.axs[0][1].plot(xs, self.right_grfs)
        plt.legend()
        time.sleep(0.5)

    def update(self, action, left_grfs, right_grfs):

        self.left_grfs = np.hstack((self.left_grfs[1:], left_grfs))
        self.right_grfs = np.hstack((self.right_grfs[1:], right_grfs))

        self.axs[0][0].set_ylim(bottom=0, top=np.max(self.left_grfs))
        self.axs[0][0].set_xlim(left=0, right=120)
        self.axs[0][1].set_ylim(bottom=0, top=np.max(self.right_grfs))
        self.axs[0][1].set_xlim(left=0, right=120)

        self.grfline1.set_ydata(self.left_grfs)
        self.grfline2.set_ydata(self.right_grfs)

        for i, s in enumerate(['left', 'right']):
            for j, l in enumerate(['hip roll', 'hip yaw', 'hip pitch', 'knee', 'foot']):
                setpoint = action[i * 5 + j]
                pgain = action[10 + i * 5 + j]
                l1 = self.data[s][l]['line1']
                l2 = self.data[s][l]['line2']
                d1 = np.hstack((self.data[s][l]['P'][1:], pgain))
                d2 = np.hstack((self.data[s][l]['setpoint'][1:], setpoint))
                l1.set_ydata(d1)
                l2.set_ydata(d2)
                self.data[s][l]['P'] = d1
                self.data[s][l]['setpoint'] = d2
                self.axs[1+j][i].set_ylim(bottom=-1, top=1)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
