# read csv
# make into matrix
# save as image using imageio
# run like this:
# python3 data_gathering/csv_to_image.py "gifs/something*" output.gif

import sys
from glob import glob

import imageio
import numpy as np
import re

def add_gridlines(im, grid_size, color):
    im[::grid_size, :] = color
    im[:, ::grid_size] = color
    return im

def numerical_sort_key(filename):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', filename)]


def main():
    files = sys.argv[1]
    output = sys.argv[2]
    size = 10
    speed = 1
    data = [
        add_gridlines(
        np.repeat(255 - np.repeat(np.loadtxt(file, dtype='u1', delimiter=',')*255, size, axis=0), size, axis=1),
        grid_size=size,
        color=80
        )
        for file in sorted(glob(files), key=numerical_sort_key)
    ]

    with imageio.get_writer(output, mode='I') as writer:
        for im in data:
            for _ in range(speed):
                writer.append_data(im)

if __name__ == '__main__':
    main()
