import os
import numpy as np
import argparse

from PIL import Image
# from matplotlib import image
# from matplotlib import pyplot as plt

DEBUG = True

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default=None, help="name of png file to convert")
parser.add_argument("--dest", type=str, default=None, help="name of npy file to write out")
parser.add_argument("--noviz", default=False, help="stop debug output", action="store_true")
args = parser.parse_args()

if args.noviz:
    DEBUG = False

# load image as pixel array
image = Image.open(args.src).convert('L')  # convert to single-channel
if DEBUG:
    print("reading as PILLOW image")
    print(image.size)
    print(image.format)
    print(image.mode)
    image.show()

# resize to (500, 500)
image = image.resize((500, 500))
if DEBUG:
    print("resizing to 500,500")
    print(image.size)
    print(image.format)
    print(image.mode)
    image.show()

# convert to numpy array
data = np.asarray(image, dtype=np.float64)
data *= 1.0/data.max()
if DEBUG:
    print("converted to np.array")
    print(type(data))
    print(data.shape)
    print(data.dtype)

# write it out
with open(args.dest, 'wb') as f:
    np.save(f, data)
