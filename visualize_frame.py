# Given an npy file of shape (num_states, H, W, 3), visualize the frame i
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--state", type=int, required=True)
args = parser.parse_args()

file = args.file
state = args.state

data = np.load(file)
frame = data[state]
plt.imshow(frame)
plt.show()
