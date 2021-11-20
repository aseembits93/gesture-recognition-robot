import numpy as np
import os
from perlin_noise import PerlinNoise

def generate_perlin(nrow, ncol):
    hgtmaps = [np.zeros((nrow, ncol))]
    for seed in range(1):
        for octave in [30, 50]:
            path = './envs/terrains/precomputed/'
            if not os.path.isdir(path):
                os.mkdir(path)
            fname = os.path.join('./envs/terrains/precomputed/', 'perlin-seed{}-octave{}.npy'.format(seed+1, octave))
            if os.path.isfile(fname):
                hgtmaps += [np.load(fname)]
            else:
                print("First-time run: generating terrain map", fname)
                fn = PerlinNoise(octaves=octave, seed=seed+1)
                height_map = [[fn([i/nrow, j/ncol]) for j in range(nrow)] for i in range(ncol)]
                height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
                hgtmaps += [height_map]
                np.save(fname, height_map)
    return hgtmaps

