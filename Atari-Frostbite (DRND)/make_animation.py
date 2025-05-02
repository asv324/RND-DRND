from matplotlib.animation import FuncAnimation
from matplotlib.animation import Animation
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from online.config_DRND import *

matplotlib.use('TkAgg')

env_id = default_config['EnvID']
with open('int_reward_{}'.format(env_id), 'rb') as f:
    pkl = pickle.load(f)

fig, ax = plt.subplots()
xdata, ydata = [], []

line, = ax.plot([], [], lw=2)


def init():
    ax.set_xlim(0, len(pkl) + 5)
    ax.set_ylim(pkl.min(), 1)
    return line,


def update(frame):
    xdata.append(int(frame) - 1)

    ydata.append(pkl[int(frame) - 1, 0])
    line.set_data(xdata, ydata)
    return line,


ani = FuncAnimation(fig, update, frames=np.linspace(0, len(pkl) - 1, len(pkl), endpoint=False),
                    init_func=init, blit=True, interval=20, repeat=False)
# plt.show()
Animation.save(ani, filename='int_reward_vid.gif')