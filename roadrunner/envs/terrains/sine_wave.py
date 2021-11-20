import numpy as np
import os

def generate_sine_wave(nrow, ncol):
    hgtmaps = [np.zeros((nrow, ncol))]
    path = './envs/terrains/precomputed/'
    if not os.path.isdir(path):
        os.mkdir(path)
    fname = os.path.join('./envs/terrains/precomputed/', 'slope.npy')
    if os.path.isfile(fname):
        hgtmaps += [np.load(fname)]
    else:
        print("First-time run: generating terrain map", fname)
        height_map = [[np.sin(20*np.pi*i/ncol + 10) for j in range(nrow)] for i in range(ncol)]
        height_map = np.transpose(height_map)
        height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
        hgtmaps += [height_map]
        np.save(fname, height_map)
    return hgtmaps
