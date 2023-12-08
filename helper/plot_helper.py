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
    title = f'$L={L}, N={N}'
    if U is not None:
        title += f', U={U}'
    if theta is not None:
        title += r', \theta ' + f'={theta/math.pi:.3f}\pi'
    title += '$'
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
    style_list = None
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
        if 'xlim' in fig_args.keys():
            ax.set_xlim(fig_args['xlim'])
        if 'ylog' in fig_args.keys():
            ax.set_yscale('log')
        if 'hline' in fig_args.keys():
            ax.axhline(fig_args['hline'], ls='--', c='gray')
        if 'hline2' in fig_args.keys():
            ax.axhline(fig_args['hline2'], ls='--', c='tomato')
        if 'filter' in fig_args.keys():
            filter = fig_args['filter']
        if 'anno' in fig_args.keys():
            ax.annotate(fig_args['anno'], xy=(0.5, 0.1), xycoords='axes fraction')
        if 'style_list' in fig_args.keys():
            style_list = fig_args['style_list']
        if 'scatter1' in fig_args.keys():
            ax.scatter(fig_args['scatter1'][0], fig_args['scatter1'][1], color='gray')
        if 'scatter2' in fig_args.keys():
            ax.scatter(fig_args['scatter2'][0], fig_args['scatter2'][1], color='gray')

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
        if style_list is None:
            style = {}
        else:
            style = style_list[i]
        if label_list is None:
            ax.plot(x_arr, y_arr, **style)
        else:
            ax.plot(x_arr, y_arr, label=label_list[i], **style)

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



def plot_twinx(fig_name, x_list, y_lists1, y_lists2, label_list, style_list,
               fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4.8, 3.5), dpi=300)
    fig.subplots_adjust(left=0.15, right=0.87, top=0.85, bottom=0.13)
    ax = fig.add_subplot(111)

    axt = ax.twinx()

    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax.set_xlabel(fig_args['xlabel'])
        if 'ylabel1' in fig_args.keys():
            ax.set_ylabel(fig_args['ylabel1'])
        if 'ylabel2' in fig_args.keys():
            axt.set_ylabel(fig_args['ylabel2'])
        if 'title' in fig_args.keys():
            ax.set_title(fig_args['title'])
        if 'ylim2' in fig_args.keys():
            axt.set_ylim(fig_args['ylim2'])
        if 'xlim' in fig_args.keys():
            ax.set_xlim(fig_args['xlim'])
        if 'ylog' in fig_args.keys():
            ax.set_yscale('log')
        if 'hline1' in fig_args.keys():
            ax.axhline(fig_args['hline1'], ls='--', c='gray')
        if 'hline2' in fig_args.keys():
            axt.axhline(fig_args['hline2'], ls='--', c='gray')
        if 'filter' in fig_args.keys():
            filter = fig_args['filter']
        if 'anno' in fig_args.keys():
            ax.annotate(fig_args['anno'], xy=(0.87, 0.935), xycoords='axes fraction')
        if 'anno2' in fig_args.keys():
            ax.annotate(fig_args['anno2'], xy=(0.02, 0.935), xycoords='axes fraction')
        if 'style_list' in fig_args.keys():
            style_list = fig_args['style_list']
        if 'scatter1' in fig_args.keys():
            ax.scatter(fig_args['scatter1'][0], fig_args['scatter1'][1], color='gray')
        if 'scatter2' in fig_args.keys():
            ax.scatter(fig_args['scatter2'][0], fig_args['scatter2'][1], color='gray')
        if 'scatter3' in fig_args.keys():
            axt.scatter(fig_args['scatter3'][0], fig_args['scatter3'][1], color='black')

    ind = -1
    l_list = []
    for y_list in y_lists1:
        ind += 1
        l1, = ax.plot(x_list, y_list, **style_list[ind])
        l_list.append(l1)

    for y_list in y_lists2:
        ind += 1
        l1, = axt.plot(x_list, y_list, **style_list[ind])
        l_list.append(l1)

    ax.legend(l_list, label_list,
        bbox_to_anchor=(0.5, 1.1), ncol=len(l_list), loc='center', fontsize=8,
        borderpad=0.2, handlelength=1.5, handletextpad=0.6, labelspacing=0.2,
        columnspacing=1, framealpha=0.5)



    plt.savefig(fig_name)
    plt.close()
    gc.collect()



#===============================================================================
def plot_scatter(fig_name, x_list, y_lists, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(dpi=600)

    legend_list = None
    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax.set_xlabel(fig_args['xlabel'])
        if 'ylabel' in fig_args.keys():
            ax.set_ylabel(fig_args['ylabel'])
        if 'title' in fig_args.keys():
            ax.set_title(fig_args['title'], pad=25)
        if 'hline' in fig_args.keys():
            ax.axhline(fig_args['hline'], ls='--', c='gray')
        if 'ylim' in fig_args.keys():
            ax.set_ylim(fig_args['ylim'][0], fig_args['ylim'][1])
        if 'xticklabels' in fig_args.keys():
            xticks_pos = np.array(x_list)
            ax.set_xticks(x_list)
            xticks_lab = np.array(fig_args['xticklabels'])
            if xticks_pos.shape[0] > 100:
                labels = [xticks_lab[i] if i%20==0 or '4' in xticks_lab[i]
                    else '' for i, item in enumerate(ax.get_xticklabels())]
                xticks_lab = labels
            ax.set_xticklabels(xticks_lab, rotation=50, ha='right', fontsize=8,
                           rotation_mode='anchor')
        if 'legend_list' in fig_args.keys():
            legend_list = fig_args['legend_list']



    color_list = ['tomato', 'cornflowerblue', 'forestgreen', 'gold', 'orange',
                  'rebeccapurple', 'slategrey', 'mediumblue', 'torquoise',
                  'maroon']
    ms_list = ['o', 'v', 'X', 's', 'D', 'P', 'h', '<', '*']

    for i, y_list in enumerate(y_lists):
        if not isinstance(x_list[0], np.float64):
            x_list = x_list[i]

        if len(y_lists) <= 10:
            col = color_list[i]
            ms = ms_list[i]
            s = None
        else:
            col = 'black'
            ms = 'o'
            s=0.5

        if legend_list is None:
            ax.scatter(x_list, y_list, c=col, marker=ms, s=s)
        else:
            ax.scatter(x_list, y_list, c=col, marker=ms, s=s, label=legend_list[i])

    if legend_list is None:
        pass
    else:
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05),
                  ncol=len(legend_list), borderpad=0.2, handlelength=0.5,
                  handletextpad=0.6, labelspacing=0.2)


    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    gc.collect()


#===============================================================================
def plot_errorbar(fig_name, x_list, y_lists, yerr_lists, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()

    ms = 2
    legend_list = None
    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax.set_xlabel(fig_args['xlabel'])
        if 'ylabel' in fig_args.keys():
            ax.set_ylabel(fig_args['ylabel'])
        if 'ylim' in fig_args.keys():
            ax.set_ylim(fig_args['ylim'])
        if 'title' in fig_args.keys():
            ax.set_title(fig_args['title'], pad=25)
        if 'hline' in fig_args.keys():
            ax.axhline(fig_args['hline'], ls='--', c='gray')
        if 'hline_list' in fig_args.keys() and fig_args['hline_list'] is not None:
            height_list, box_list = fig_args['hline_list']
            box_list = [el for el in box_list]
            lx = len(x_list)
            xpos1 = 0
            for i, h in enumerate(height_list):
                xpos2 = xpos1 + box_list[i]/lx
                ax.axhline(h, xpos1, xpos2, ls='-', c='limegreen', lw=2, zorder=100)
                ax.axhline(h, xpos1, xpos2, ls=':', c='black', lw=2, zorder=100)
                xpos1 += box_list[i]/lx
            ax.axhline(h, xpos1, xpos2, ls='-', c='limegreen', lw=2, zorder=100)
            ax.axhline(h, xpos2, 1, ls=':', c='black', lw=2, zorder=100)
            ax.set_xlim(min(x_list), max(x_list))
        if 'xticklabels' in fig_args.keys():
            xticks_pos = np.array(x_list)
            ax.set_xticks(x_list)
            xticks_lab = np.array(fig_args['xticklabels'])
            if xticks_pos.shape[0] > 100:
                labels = [xticks_lab[i] if i%20==0 else ''
                          for i, item in enumerate(ax.get_xticklabels())]
                # labels = [xticks_lab[i] if i%20==0 or '4' in xticks_lab[i]
                #     else '' for i, item in enumerate(ax.get_xticklabels())]
                xticks_lab = labels
            ax.set_xticklabels(xticks_lab, rotation=50, ha='right', fontsize=8,
                           rotation_mode='anchor')
        if 'legend_list' in fig_args.keys():
            legend_list = fig_args['legend_list']
        if 'ms' in fig_args.keys():
            ms = fig_args['ms']



    assert len(y_lists) <= 10
    color_list = ['tomato', 'cornflowerblue', 'forestgreen', 'gold', 'orange',
                  'rebeccapurple', 'slategrey', 'mediumblue', 'torquoise',
                  'maroon']
    ms_list = ['o', 'v', 'X', 's', 'D', 'P', 'h', '<', '*']

    for i, y_list in enumerate(y_lists):
        print(y_list)

        if legend_list is None:
            ax.errorbar(x_list, y_list, yerr=yerr_lists[i], c=color_list[i],
                        fmt=ms_list[i], ms=ms)
        else:
            ax.errorbar(x_list, y_list, yerr=yerr_lists[i], c=color_list[i],
                        label=legend_list[i], fmt=ms_list[i], ms=ms)

    if legend_list is None:
        pass
    else:
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05),
                  ncol=len(legend_list), borderpad=0.2, handlelength=0.5,
                  handletextpad=0.6, labelspacing=0.2)


    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    gc.collect()


#===============================================================================
def polar_corrd(fig_name, time, abs_vals, angle_vals, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.3*0.6, 2), dpi=300)
    fig.subplots_adjust(bottom=0.2, right=.97, top=0.85, left=0.11)
    canv = gridspec.GridSpec(2, 1, height_ratios=[0.05, 1], hspace=0.1)
    canv_bot = gridspec.GridSpecFromSubplotSpec(1, 2, canv[1, 0],
        width_ratios=[1, 1], wspace=0.3)

    ax1 = plt.subplot(canv_bot[0, 0])
    ax2 = plt.subplot(canv_bot[0, 1])
    axl = plt.subplot(canv[0, 0])
    axl.set_axis_off()

    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax1.set_xlabel(fig_args['xlabel'])
            ax2.set_xlabel(fig_args['xlabel'])
        if 'label_dum' in fig_args.keys():
            label_dum = fig_args['label_dum']
            ax1.set_ylabel('$abs( ' + label_dum +' )$')
            ax2.set_ylabel('$angle( ' + label_dum +' )/\pi$', labelpad=0)
        if 'title' in fig_args.keys():
            axl.set_title(fig_args['title'])
        if 'hline' in fig_args.keys():
            [ax2.axhline(hl/np.pi, ls='-', c='black', lw=0.5)
                for hl in fig_args['hline']]

    ax1.axhline(1, ls='--', c='gray')
    ax1.set_ylim((0, 1.05))
    ax1.set_xlim((0, time[-1]))
    ax2.set_xlim((0, time[-1]))
    # print(abs_vals)
    ax1.plot(time, abs_vals, c='tomato')
    ax2.scatter(time, angle_vals/np.pi, c='cornflowerblue', marker='o', s=1)

    plt.savefig(fig_name)
    plt.close()
    gc.collect()


#===============================================================================
def complx_plane(fig_name, cmplx_exp, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))

    legend_list = None
    if fig_args is not None:
        if 'obs' in fig_args.keys():
            obs_name = fig_args['obs']
            ax.set_xlabel('$Re(' + obs_name + ')$')
            ax.set_ylabel('$Im(' + obs_name + ')$')
        if 'title' in fig_args.keys():
            ax.set_title(fig_args['title'], pad=25)
        if 'unit_circle' in fig_args.keys() and fig_args['unit_circle']:
            circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None')
            ax.add_patch(circ)
        if 'scatter' in fig_args.keys():
            ax.scatter(fig_args['scatter'].real, fig_args['scatter'].imag,
                       c='red')


    ax.set_aspect('equal')
    ax.plot(cmplx_exp.real, cmplx_exp.imag, color='cornflowerblue')


    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    gc.collect()


#===============================================================================
# colorplots
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



def num_op_2cplot(fig_name, time, L, mat1, mat2, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.3/1.7, 2.8), dpi=300)
    fig.subplots_adjust(bottom=0.18, right=.92, top=0.88, left=0.10)
    canv = gridspec.GridSpec(2, 1, height_ratios=[0.05, 1], hspace=0.35)
    canv_bot = gridspec.GridSpecFromSubplotSpec(1, 2, canv[1, 0],
        width_ratios=[2, 0.05], wspace=0.05)
    canv_grid = gridspec.GridSpecFromSubplotSpec(1, 2, canv_bot[0, 0],
        width_ratios=[1, 1], wspace=0.2)


    ax1 = plt.subplot(canv_grid[0, 0])
    ax2 = plt.subplot(canv_grid[0, 1])
    axleg = plt.subplot(canv_bot[0, 1])
    axl = plt.subplot(canv[0, 0])
    axl.set_axis_off()

    vmin = min(np.min(mat1), np.min(mat2))
    vmax = 0.5 #max(np.max(mat1), np.max(mat2))
    x, y = np.meshgrid(range(1, L+1), time)
    im = ax1.pcolormesh(x, y, mat1.T, cmap='Blues', shading='nearest', vmin=vmin, vmax=vmax)
    im = ax2.pcolormesh(x, y, mat2.T, cmap='Blues', shading='nearest', vmin=vmin, vmax=vmax)

    ax1.set_ylabel('$t$', fontsize=14)

    for ax in [ax1, ax2]:
        # [ax.axvline(i+0.5, c='black', lw=0.5, alpha=0.5) for i in range(1, L)]
        ax.tick_params(labelsize=10)

        ax.set_xticks(np.arange(1, L+1, 1))
        ax.tick_params(axis="x", direction="in", color='orange')
        if L <= 8:
            ax.set_xticks(list(range(1, L+1)))
        else:
            labs = list(range(1, L+1, int(L/6)))
            ax.set_xticklabels([i if i in labs else '' for i in range(1, L+1)])

        ax.set_xlabel('site $i$', fontsize=14)


    if fig_args is not None:
        if 'title1' in fig_args.keys():
            ax1.set_title(fig_args['title1'])
        if 'title2' in fig_args.keys():
            ax2.set_title(fig_args['title2'])
        if 'main_title' in fig_args.keys():
            axl.set_title(fig_args['main_title'])

    cbar = fig.colorbar(im,  cax=axleg)
    cbar.ax.tick_params(labelsize=12)
    path_fig = fig_name.resolve()
    plt.savefig(path_fig)
    plt.close()
    gc.collect()
    # if bool_convert_trim():
    #     subprocess.call(['convert', path_fig, '-trim', path_fig])



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
    fig = plt.figure(figsize=(8.3*0.6, 4), dpi=300)
    ax = fig.add_subplot(111)


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
        if 'cmap' in fig_args.keys():
            cmap = fig_args['cmap']
        if 'shading' in fig_args.keys():
            shading = fig_args['shading']
        if 'lognorm' in fig_args.keys():
            norm = colors.LogNorm(vmin=1e-3, vmax=1)

    x, y = np.meshgrid(xarr, yarr)
    im = ax.pcolormesh(x, y, mat, cmap=cmap, norm=norm, shading=shading)
    cbar = plt.colorbar(im)

    if fig_args is not None:
        if 'clabel' in fig_args.keys():
            cbar.ax.set_ylabel(fig_args['clabel'], fontsize=10)

    plt.savefig(fig_name)
    plt.close(fig)
    gc.collect()



def make_4cplots_time(fig_name, time, L, mat_list, fig_args=None):
    fig_name.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.3, 2), dpi=300)
    fig.subplots_adjust(bottom=0.18, right=.92, top=0.85, left=0.06)
    canv = gridspec.GridSpec(2, 1, height_ratios=[0.05, 1], hspace=0.2)
    canv_bot = gridspec.GridSpecFromSubplotSpec(1, 4, canv[1, 0],
        width_ratios=[1, 1, 1, 1], wspace=0.3)

    ax1 = plt.subplot(canv_bot[0, 0])
    ax2 = plt.subplot(canv_bot[0, 1])
    ax3 = plt.subplot(canv_bot[0, 2])
    ax4 = plt.subplot(canv_bot[0, 3])
    axl = plt.subplot(canv[0, 0])
    axl.set_axis_off()

    if fig_args is not None:
        if 'xlabel' in fig_args.keys():
            ax1.set_xlabel(fig_args['xlabel'])
            ax2.set_xlabel(fig_args['xlabel'])
            ax3.set_xlabel(fig_args['xlabel'])
            ax4.set_xlabel(fig_args['xlabel'])
        if 'ylabel' in fig_args.keys():
            ax1.set_ylabel(fig_args['ylabel'])
        if 'title_list' in fig_args.keys():
            title_list = fig_args['title_list']
            ax1.set_title(title_list[0], fontsize=10)
            ax2.set_title(title_list[1], fontsize=10)
            ax3.set_title(title_list[2], fontsize=10)
            ax4.set_title(title_list[3], fontsize=10)
        if 'main_title' in fig_args.keys():
            axl.set_title(fig_args['main_title'])

    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])

    x, y = np.meshgrid(range(L), time)

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):

        im = ax.pcolormesh(x, y, mat_list[i].T, cmap='turbo', shading='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="9%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        ax.set_xticklabels([])
        if i > 0:
            ax.set_yticklabels([])


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
        ax.set_xticks(np.arange(1, L+1))
        ax.set_yticks(np.arange(1, L+1))
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
