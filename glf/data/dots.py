# https://github.com/ermongroup/BiasAndGeneralization/blob/master/DotsAndPie/dataset/generate/dots_generator.py

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def gen_image_count(num_object=3, overlap=False):
    radius = 0.08
    while True:
        shifts = np.random.uniform(radius, 1.0-radius, size=(num_object, 2))
        dist1 = np.tile(np.expand_dims(shifts, axis=0), (num_object, 1, 1))
        dist2 = np.tile(np.expand_dims(shifts, axis=1), (1, num_object, 1))
        dist = np.sqrt(np.sum(np.square(dist1 - dist2), axis=2))
        np.fill_diagonal(dist, 1.0)
        if not overlap and np.min(dist) > 2.1 * radius:
            break
        if overlap and np.min(dist) > 2 * radius * 0.9:
            break

    margin = 5
    fig = plt.figure(figsize=((64+2*margin)/10.0, (64+2*margin)/10.0), dpi=10)
    ax = plt.gca()
    for i in range(num_object):
        random_color = np.random.uniform(0, 0.9, size=(3,))
        circle = plt.Circle(shifts[i], radius, color=random_color)
        ax.add_artist(circle)
    plt.axis('off')
    plt.tight_layout()

    arr = fig2data(fig)
    arr = arr[margin:64+margin, margin:64+margin, :3]

    plt.close(fig)
    return arr