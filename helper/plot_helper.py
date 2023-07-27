import os
import gc
import math
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable



def bool_convert_trim():
    if shutil.which('qstat') is None:
        plt.rc('text', usetex=True)
        return True
    else:
        plt.rc('text', usetex=False)
        return False


def make_title(L, N, U, theta, psi0=None):
    title = f'$L={L}, N={N}, U={U}, ' + r'\theta ' + f'={theta/math.pi:.3f}\pi$'
    if psi0 is not None:
        return title + f', {psi0}'
    else:
        return title



class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


#===============================================================================
def plot_lines(fig_name, x_lists, y_lists, label_list=None, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()

    tcut = None  # default
    filter = None
    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax.set_xlabel(fig_args['xlabel'])
        if 'ylabel' in fig_args.keys():
            ax.set_ylabel(fig_args['ylabel'])
        if 'title' in fig_args.keys():
            ax.set_title(fig_args['title'])
        if 'tcut' in fig_args.keys():
            tcut = fig_args['tcut']
        if 'ylim' in fig_args.keys():
            ax.set_ylim(fig_args['ylim'])
        if 'ylog' in fig_args.keys():
            ax.set_yscale('log')
        if 'hline' in fig_args.keys():
            ax.axhline(fig_args['hline'], ls='--', c='gray')
        if 'filter' in fig_args.keys():
            filter = fig_args['filter']

    empty = True
    for i, y_list in enumerate(y_lists):
        y_arr = np.array(y_list)
        if filter is not None and np.max(np.abs(y_arr)) < filter:
            continue
        empty = False
        if len(x_lists) == 1:
            x_list = x_lists[0]
        else:
            x_list = x_lists[i]
        x_arr = np.array(x_list)
        if tcut is not None:
            ind = np.abs(time-tcut).argmin()
            x_arr = x_arr[:ind]
            y_arr = y_arr[:ind]
        if label_list is None:
            ax.plot(x_arr, y_arr)
        else:
            ax.plot(x_arr, y_arr, label=label_list[i])

    if empty:
        plt.close()
        return

    if label_list is None:
        pass
    elif 10 <= len(label_list):
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend()
    plt.savefig(fig_name)
    plt.close()
    gc.collect()


#===============================================================================
def plot_histogram(fig_name, x_vals, y_vals, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax.set_xlabel(fig_args['xlabel'])
        if 'ylabel' in fig_args.keys():
            ax.set_ylabel(fig_args['ylabel'])
        if 'title' in fig_args.keys():
            ax.set_title(fig_args['title'])


    # ax.hist(x=counts, bins=bins, histtype='bar', density=True)
    ax.set_xticks(x_vals)
    ax.scatter(x_vals, y_vals)

    plt.savefig(fig_name)
    plt.close()
    gc.collect()


#===============================================================================
def num_op_cplot(fig_name, time, L, num_op_mat, title):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    x, y = np.meshgrid(range(1, L+1), time)
    im = ax.pcolormesh(x, y, np.transpose(num_op_mat), cmap='turbo',
        shading='nearest')
    [ax.axvline(i+0.5, c='black', lw=2) for i in range(1, L)]
    ax.tick_params(labelsize=12)
    ax.set_xticks(list(range(1, L+1)))
    ax.set_xlabel('site $i$', fontsize=14)
    ax.set_ylabel('time', fontsize=14)
    ax.set_title(title)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(r"$\langle \Psi|\hat{n}_i| \Psi\rangle$",
        fontsize=14)
    # plt.show()
    path_fig = fig_name.resolve()
    plt.savefig(path_fig)
    plt.close()
    gc.collect()
    if bool_convert_trim():
        subprocess.call(['convert', path_fig, '-trim', path_fig])


# def plot_ninj_mat_time(self, t_instance):
#     fig, ax = plt.subplots()
#     L = self.L
#     ax.set_xticks(range(L))
#     ax.set_yticks(range(L))
#     ax.tick_params(axis="both", direction="in", color='orange')
#     ax.yaxis.set_ticks_position('both')
#     ax.xaxis.set_ticks_position('both')
#
#     _, _, ninj_mat = self.get_ninj_mat_time(t_instance)
#     x, y = np.meshgrid(range(L+1), range(L+1))
#
#     im = ax.pcolormesh(x, y, ninj_mat.T, cmap='viridis',#, vmin=0, vmax=2,
#         shading='flat')
#     ax.set_xlabel('site i', fontsize=14)
#     ax.set_ylabel('site j', fontsize=14)
#     cbar = fig.colorbar(im)
#     cbar.ax.tick_params(labelsize=12)
#     cbar.ax.set_ylabel(r"$\langle \Psi|n_i n_j| \Psi\rangle$",
#         fontsize=14)
#     plt.show()
#     # path_fig = (self.path_prop/fig_name).resolve()
#     # plt.savefig(path_fig)
#     # plt.close()
#     # if self.bool_convert_trim:
#     #     subprocess.call(['convert', path_fig, '-trim', path_fig])


#===============================================================================
def make_cplot(fig_name, xarr, yarr, mat, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()

    cmap = 'turbo'  # defualt
    shading = 'nearest'
    norm = None     # defualt
    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax.set_xlabel(fig_args['xlabel'])
        if 'ylabel' in fig_args.keys():
            ax.set_ylabel(fig_args['ylabel'])
        if 'title' in fig_args.keys():
            ax.set_title(fig_args['title'])
        if 'cmap_div' in fig_args.keys():
            norm = MidpointNormalize(vmin=np.min(mat), vmax=np.max(mat),
                midpoint=0)
            cmap = 'seismic'
        if 'shading' in fig_args.keys():
            shading = fig_args['shading']

    x, y = np.meshgrid(xarr, yarr)
    im = ax.pcolormesh(x, y, mat, cmap=cmap, norm=norm, shading=shading)
    cb = plt.colorbar(im)

    plt.savefig(fig_name)
    plt.close(fig)
    gc.collect()



def make_5cplots(fig_name, t_list, L, mat_list, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.3, 2), dpi=300)
    fig.subplots_adjust(bottom=0.18, right=.92, top=0.85, left=0.04)
    canv = gridspec.GridSpec(2, 1, height_ratios=[0.05, 1], hspace=0.2)
    canv_bot = gridspec.GridSpecFromSubplotSpec(1, 5, canv[1, 0],
        width_ratios=[1, 1, 1, 1, 1], wspace=0.3)

    ax1 = plt.subplot(canv_bot[0, 0], aspect=1)
    ax2 = plt.subplot(canv_bot[0, 1], aspect=1)
    ax3 = plt.subplot(canv_bot[0, 2], aspect=1)
    ax4 = plt.subplot(canv_bot[0, 3], aspect=1)
    ax5 = plt.subplot(canv_bot[0, 4], aspect=1)
    axl = plt.subplot(canv[0, 0])
    axl.set_axis_off()

    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax1.set_xlabel(fig_args['xlabel'])
            ax2.set_xlabel(fig_args['xlabel'])
            ax3.set_xlabel(fig_args['xlabel'])
            ax4.set_xlabel(fig_args['xlabel'])
            ax5.set_xlabel(fig_args['xlabel'])
        if 'ylabel' in fig_args.keys():
            ax1.set_ylabel(fig_args['ylabel'])
        if 'title_list' in fig_args.keys():
            title_list = fig_args['title_list']
            ax1.set_title(title_list[0], fontsize=10)
            ax2.set_title(title_list[1], fontsize=10)
            ax3.set_title(title_list[2], fontsize=10)
            ax4.set_title(title_list[3], fontsize=10)
            ax5.set_title(title_list[4], fontsize=10)
        if 'main_title' in fig_args.keys():
            axl.set_title(fig_args['main_title'])

    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])
    ax5.set_yticklabels([])

    x, y = np.meshgrid(range(L+1), range(L+1))

    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):

        im = ax.pcolormesh(x, y, mat_list[i].T, cmap='viridis', shading='flat')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="9%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        ax.tick_params(axis="both", direction="in", color='orange')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')


    if fig_args is not None:
        if 'clim1' in fig_args.keys():
            im1.set_clim(fig_args['clim1'])
        if 'clabel' in fig_args.keys():
            cbar.ax.set_ylabel(fig_args['clabel'], fontsize=10)


    plt.savefig(fig_name)
    plt.close(fig)
    gc.collect()

    # path_cwd = os.getcwd()
    # os.chdir(fig_name.parent)
    #
    # if fig_name.name[-3:] == 'png':
    #     subprocess.check_output(["convert", fig_name.name, "-trim", fig_name.name])
    # elif fig_name.name[-3:] == 'pdf':
    #     subprocess.check_output(["pdfcrop", fig_name.name, fig_name.name])
    #
    # os.chdir(path_cwd)
