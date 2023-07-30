import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.colors as mcolors
from random import sample

ALL_COLORS = list(mcolors.XKCD_COLORS.keys())


def get_node_trajectories(rollout_array, n_node):
    """Split batch rollout into individual rollouts

    :param rollout_array: T+1 x N x _ array of the system trajectories
    :param n_node: array [B] (graph.n_node) containing the number of nodes in each graph in a batch.
    :return: list of individual system trajectories. List is size B.
    """
    return np.split(rollout_array, n_node, axis=1)[:-1]


def animate_trajectory(rollout, system_size, title=None, outpath='movie', fps=5):
    """Animate two trajectory of a rollout

    Nodes colore are chosen at random and do NOT (currently) scale with their size or interactions, i.e. all atoms
    are the same size and shape but differ in a randomly selected color.

    :param rollout: array of T x N x 2 (num_steps x num_atoms x [pos_x, pos_y]
    :param system_size: array([[xlo, ylo],[xhi, yhi]])
    :param title: str, corresponding to the title of the graph
    :param outpath: str of outpath without the suffic (without .mp4)
    :param fps: int; frames per second
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    interval = 1000/fps
    xlo, ylo = system_size[0, :]
    xhi, yhi = system_size[1, :]
    lx, ly = xhi-xlo, yhi-ylo
    hist_length = 8
    ax.set_xlim(-4-.1*lx, 4+.1*lx)
    ax.set_ylim(-4-.1*ly, 4+.1*ly)
    ax.plot([xlo, xhi], [yhi, yhi], 'k-', lw=4)
    ax.plot([xlo, xhi], [ylo, ylo], 'k-', lw=4)
    ax.plot([xlo, xlo], [ylo, yhi], 'k-', lw=4)
    ax.plot([xhi, xhi], [ylo, yhi], 'k-', lw=4)
    lines, hists = [], []
    num_steps, num_atoms = rollout.shape[:2]
    colors = sample(ALL_COLORS, num_atoms)
    prev_seq_mask = np.arange(0 - hist_length, 1, dtype=np.int32)
    prev_seq_mask[prev_seq_mask < 0] = 0
    if title:
        ax.set_title(title, fontsize=24)

    for atom_ind in range(num_atoms):
        prev_seq = rollout[prev_seq_mask, atom_ind, :2]
        hist, = ax.plot(prev_seq[:, 0], prev_seq[:, 1], lw=32, color=colors[atom_ind], alpha=0.5)
        hist.set_solid_capstyle('round')

        line, = ax.plot([rollout[0, atom_ind, 0]], [rollout[0, atom_ind, 1]],
                        'o', markersize=32, markeredgewidth=4, markeredgecolor='k', color=colors[atom_ind])
        lines.append(line)
        hists.append(hist)

    def animate(i):
        prev_seq_mask = np.arange(i - hist_length, i + 1, dtype=np.int32)
        prev_seq_mask[prev_seq_mask < 0] = 0
        for idx, line in enumerate(lines):
            prev_seq = rollout[prev_seq_mask, idx, :2]

            hists[idx].set_data(prev_seq[:, 0], prev_seq[:, 1])

            line.set_data([rollout[i, idx, 0]], [rollout[i, idx, 1]])
            ax.set_title(f'Time step: {i}')

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=num_steps,
                                             interval=interval, repeat=True)
    writer = matplotlib.animation.FFMpegWriter(fps=fps)
    ani.save(f'{outpath}'+'.mp4', writer=writer)
    plt.close()


def animate_two_trajectory(rollout1, rollout2, system_size, titles=None, outpath='movie', fps=5):
    """animate two trajectories side by side

    Animates the trajectories of two rollouts side by side. Meant to compare the trajectories from the simulator and
    the model. A history of 8 previous states are included as a trail behind the current state. Colors are randomly
    assigned.

    :param rollout1: array of T x N x 2 (num_steps x num_atoms x [pos_x, pos_y]
    :param rollout2: array of T x N x 2 (num_steps x num_atoms x [pos_x, pos_y]
    :param system_size: array([[xlo, ylo],[xhi, yhi]])
    :param titles: list of size 2, corresponding to the titles of each graph
    :param outpath: str of outpath without the suffic (without .mp4)
    :param fps: int; frames per second
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    interval = 1000/fps
    xlo, ylo = system_size[0, :]
    xhi, yhi = system_size[1, :]
    lx, ly = xhi-xlo, yhi-ylo
    hist_length = 8

    for idx, a in enumerate(ax):
        a.set_xlim(xlo-.1*lx, xhi+.1*lx)
        a.set_ylim(ylo-.1*ly, yhi+.1*ly)

        a.plot([xlo, xhi], [yhi, yhi], 'k-', lw=4)
        a.plot([xlo, xhi], [ylo, ylo], 'k-', lw=4)
        a.plot([xlo, xlo], [ylo, yhi], 'k-', lw=4)
        a.plot([xhi, xhi], [ylo, yhi], 'k-', lw=4)

    lines1, lines2, hists1, hists2 = [], [], [], []
    num_steps, num_atoms = rollout1.shape[:2]
    colors = sample(ALL_COLORS, num_atoms)
    prev_seq_mask = np.arange(0 - hist_length, 1, dtype=np.int32)
    prev_seq_mask[prev_seq_mask < 0] = 0
    for atom_ind in range(num_atoms):
        prev_seq = rollout1[prev_seq_mask, atom_ind, :2]
        hist1, = ax[0].plot(prev_seq[:, 0], prev_seq[:, 1], lw=32, color=colors[atom_ind], alpha=0.5)
        hist1.set_solid_capstyle('round')

        prev_seq = rollout2[prev_seq_mask, atom_ind, :2]
        hist2, = ax[1].plot(prev_seq[:, 0], prev_seq[:, 1], lw=32, color=colors[atom_ind], alpha=0.5)
        hist2.set_solid_capstyle('round')

        line1, = ax[0].plot([rollout1[0, atom_ind, 0]], [rollout1[0, atom_ind, 1]],
                         'o', markersize=32, markeredgewidth=4, markeredgecolor='k', color=colors[atom_ind])
        line2, = ax[1].plot([rollout2[0, atom_ind, 0]], [rollout2[0, atom_ind, 1]],
                         'o', markersize=32, markeredgewidth=4, markeredgecolor='k', color=colors[atom_ind])

        lines1.append(line1)
        lines2.append(line2)
        hists1.append(hist1)
        hists2.append(hist2)

    def animate(i):
        prev_seq_mask = np.arange(i - hist_length, i + 1, dtype=np.int32)
        prev_seq_mask[prev_seq_mask < 0] = 0
        for idx, line in enumerate(lines1):
            prev_seq1 = rollout1[prev_seq_mask, idx, :2]
            prev_seq2 = rollout2[prev_seq_mask, idx, :2]

            hists1[idx].set_data(prev_seq1[:, 0], prev_seq1[:, 1])
            hists2[idx].set_data(prev_seq2[:, 0], prev_seq2[:, 1])

            lines1[idx].set_data([rollout1[i, idx, 0]], [rollout1[i, idx, 1]])
            lines2[idx].set_data([rollout2[i, idx, 0]], [rollout2[i, idx, 1]])

            title1 = f'{titles[0]} - step:{i}' if titles else f'step: {i}'
            title2 = f'{titles[1]} - step:{i}' if titles else f'step: {i}'
            ax[0].set_title(title1)
            ax[1].set_title(title2)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=num_steps,
                                             interval=interval, repeat=True)
    writer = matplotlib.animation.FFMpegWriter(fps=fps)
    ani.save(f'{outpath}'+'.mp4', writer=writer)
    plt.close()


def animate_trajectory_with_acc(rollout1, rollout2, system_size, outpath='movie', fps=5):
    """animate trajectory with acceleration vectors from two simulators

    Animates the trajectory of a rollout (rollout1) with acceleration vectors from both rollout1 and rollout2. It is
    assumed that the accelerations were updated in rollout2 at each timestep in rollout1. This is done to visualize
    the comparison between the model and the simulator without error accumulation.

    :param rollout1: array of T x N x 4 (num_steps x num_atoms x [pos_x, pos_y, acc_x, acc_y]
    :param rollout2: array of T x N x 4 (num_steps x num_atoms x [pos_x, pos_y, acc_x, acc_y]
    :param system_size: array([[xlo, ylo],[xhi, yhi]])
    :param outpath: str of outpath without the suffic (without .mp4)
    :param fps: int; frames per second
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    interval = 1000/fps
    xlo, xhi = system_size[:, 0]
    ylo, yhi = system_size[:, 1]
    lx, ly = xhi - xlo, yhi - ylo

    ax.set_xlim(xlo-.1*lx, xhi+.1*lx)
    ax.set_ylim(ylo-.1*ly, yhi+.1*ly)
    ax.plot([xlo, xhi], [yhi, yhi], 'k-', lw=4)
    ax.plot([xlo, xhi], [ylo, ylo], 'k-', lw=4)
    ax.plot([xlo, xlo], [ylo, yhi], 'k-', lw=4)
    ax.plot([xhi, xhi], [ylo, yhi], 'k-', lw=4)

    lines, sim_lines, nodes = [], [], []
    num_steps, num_atoms = rollout1.shape[:2]
    colors = sample(ALL_COLORS, num_atoms)
    for atom_ind in range(num_atoms):
        # set node position from model
        node, = ax.plot(
            [rollout1[0, atom_ind, 0]], [rollout1[0, atom_ind, 1]],
            'o', markersize=32, markeredgewidth=4.0, markeredgecolor='k', color=colors[atom_ind])
        nodes.append(node)

        # set acceleration vector from model
        accx, accy = rollout1[0, atom_ind, -2], rollout1[0, atom_ind, -1]
        line = ax.arrow(rollout1[0, atom_ind, 0], rollout1[0, atom_ind, 1],
                        accx, accy, facecolor=colors[atom_ind], edgecolor='xkcd:green',
                        width=0.05, linewidth=3.0, head_length=1.4*0.05, alpha=0.4)
        lines.append(line)

        # set acceleration vector from simulator
        accx, accy = rollout2[0, atom_ind, -2], rollout2[0, atom_ind, -1]
        s_line = ax.arrow(rollout1[0, atom_ind, 0], rollout1[0, atom_ind, 1],
                          accx, accy, facecolor=colors[atom_ind], edgecolor='xkcd:red',
                          width=0.05, linewidth=3.0, head_length=1.4*0.05, alpha=0.4)
        sim_lines.append(s_line)

    def animate(i):
        for idx, line in enumerate(sim_lines):
            # update nodes
            nodes[idx].set_data([rollout1[i, idx, 0]], [rollout1[i, idx, 1]])

            # update model velocity vectors
            accx, accy = rollout1[i, idx, -2], rollout1[i, idx, -1]
            lines[idx].set_data(x=rollout1[i, idx, 0], y=rollout1[i, idx, 1], dx=accx, dy=accy)

            # update simulated velocity vectors
            accx, accy = rollout2[i, idx, -2], rollout2[i, idx, -1]
            sim_lines[idx].set_data(x=rollout1[i, idx, 0], y=rollout1[i, idx, 1], dx=accx, dy=accy)

            ax.set_title(f'step: {i}')

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=num_steps,
                                             interval=interval, repeat=True)
    writer = matplotlib.animation.FFMpegWriter(fps=fps)
    ani.save(f'{outpath}'+'.mp4', writer=writer)
    plt.close()


def plot_initial_state(nodes, system_size, outpath=None):
    """Plot the initial positions and velocity vectors for a system

    :param nodes: np.array(shape=[num_atoms x 4]) where each row contains position and velocity in 2D
    :param system_size: np.array([[xlo, ylo][xhi, yhi]])
    :param outpath: str; default show, else save
    :return: None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    xlo, xhi = system_size[:, 0]
    ylo, yhi = system_size[:, 1]
    lx, ly = xhi - xlo, yhi - ylo
    ax.set_xlim(xlo - .1 * lx, xhi + .1 * lx)
    ax.set_ylim(ylo - .1 * ly, yhi + .1 * ly)

    for atom_ind in range(nodes.shape[0]):
        # set node position
        ax.plot([nodes[atom_ind, 0]], [nodes[atom_ind, 1]],
                'o', markersize=16, markeredgewidth=1.5, markeredgecolor='k')

        # set velocity vector
        velx, vely = nodes[atom_ind, 2], nodes[atom_ind, 3]
        ax.arrow(nodes[atom_ind, 0], nodes[atom_ind, 1],
                 velx, vely, width=0.04, head_length=1.4 * 0.04)

        # draw the box
        ax.plot([xlo, xhi], [yhi, yhi], 'k-', lw=4)
        ax.plot([xlo, xhi], [ylo, ylo], 'k-', lw=4)
        ax.plot([xlo, xlo], [ylo, yhi], 'k-', lw=4)
        ax.plot([xhi, xhi], [ylo, yhi], 'k-', lw=4)

    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_losses(loss1, loss2, ax):
    """Convenience function for plotting energies of a simulation

    Plots the losses of a model over T timesteps. The total rollout loss (loss1) is plotted along the primary y axis
    (left), and the one-step lines is plotted along a secondary y axis (right).

    :param loss1: array of size T: rollout losses
    :param loss2: array of size T: one-step losses
    :param ax: Matplotlib.axes.Axes object
    :return: line: matplotlib.lines.Line2D object from primary y axis
    :return: line2: matplotlib.lines.Line2D object from secondary y axis
    :return: ax2: matplotlib.axes.Axes object created from secondary y axis
    """
    ax2 = ax.twinx()

    num_steps = loss1.shape[0]
    colors = ['xkcd:blue', 'xkcd:orange']

    line, = ax.plot(np.arange(num_steps), loss1, c=colors[0])
    line2, = ax2.plot(np.arange(num_steps), loss2, c=colors[1])

    ax.set_xlabel('step', fontsize=20)
    ax.set_ylabel('Rollout loss', color=colors[0], fontsize=20)
    ax2.set_ylabel('One-step loss', color=colors[1], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15, direction='in')
    ax2.tick_params(axis='both', which='major', labelsize=15, direction='in')

    return line, line2, ax2


def plot_energies(energies1, ax, energies2=None):
    """Convenience function for plotting energies of a simulation

    Plots the kinetic, potential, and total energy of a system over T timeseps. A secondary set of energies can be
    supplied and will be plotted as a dashed line.

    :param energies1: T x 2 numpy array [pe, ke]
    :param ax: Matplotlib.axes.Axes object
    :param energies2: T x 2 numpy array [pe, ke]
    :return: matplotlib.lines.Line2D object
    """
    colors = ['#003f5c', '#bc5090', '#ffa600']
    steps = np.arange(0, energies1.shape[0])
    line, = ax.plot(steps, energies1[:, 0], c=colors[0], label='PE')
    ax.plot(steps, energies1[:, 1], c=colors[1], label='KE')
    ax.plot(steps, energies1[:, 0]+energies1[:, 1], c=colors[2], label='Etot')

    if energies2 is not None:
        ax.plot(steps, energies2[:, 0], linestyle='dashed', c=colors[0],)
        ax.plot(steps, energies2[:, 1], linestyle='dashed', c=colors[1],)
        ax.plot(steps, energies2[:, 0] + energies2[:, 1], linestyle='dashed', c=colors[2],)

    ax.set_xlabel('steps', fontsize=20)
    ax.set_ylabel('energy', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15, direction='in')
    ax.legend()

    return line
