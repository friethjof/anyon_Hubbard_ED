import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.colors import ListedColormap

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate import Propagation

import path_dirs
from helper import other_tools


def make_plot(L, N, U, psi0_str):

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    bc_name = 'obc'
    U_ = f'{U:.1f}'.replace('.', '_')
    fig_name = f'two_site_correlation_{bc_name}_L{L}_N{N}_U_{U_}.pdf'

    if bc_name == 'obc':
        bc = 'open'
    elif bc_name == 'pbc':
        bc = 'periodic'

    time_list = [4, 3, 2, 1]

    prop_class_list = []
    for theta in [el*np.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]:
        #-----------------------------------------------------------------------
        # get propoagation
        prop_class = Propagation(
            bc=bc,
            L=L,
            N=N,
            J=1,
            U=U,
            theta=theta,
            psi0_str=psi0_str,
            Tprop=5.0,
            dtprop=0.1
        )


        prop_class_list.append(prop_class)






    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3, 6), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.93, top=0.86, bottom=0.19)

    #split vertical

    canv = gridspec.GridSpec(1, 2, width_ratios=[1, 0.02], wspace=0.05)

    canv_grid = gridspec.GridSpecFromSubplotSpec(4, 5, canv[0, 0],
        width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1, 1], hspace=0.2, wspace=0.1)


    ax1_t1 = plt.subplot(canv_grid[0, 0], aspect=1)
    ax1_t2 = plt.subplot(canv_grid[1, 0], aspect=1)
    ax1_t3 = plt.subplot(canv_grid[2, 0], aspect=1)
    ax1_t4 = plt.subplot(canv_grid[3, 0], aspect=1)

    ax2_t1 = plt.subplot(canv_grid[0, 1], aspect=1)
    ax2_t2 = plt.subplot(canv_grid[1, 1], aspect=1)
    ax2_t3 = plt.subplot(canv_grid[2, 1], aspect=1)
    ax2_t4 = plt.subplot(canv_grid[3, 1], aspect=1)

    ax3_t1 = plt.subplot(canv_grid[0, 2], aspect=1)
    ax3_t2 = plt.subplot(canv_grid[1, 2], aspect=1)
    ax3_t3 = plt.subplot(canv_grid[2, 2], aspect=1)
    ax3_t4 = plt.subplot(canv_grid[3, 2], aspect=1)

    ax4_t1 = plt.subplot(canv_grid[0, 3], aspect=1)
    ax4_t2 = plt.subplot(canv_grid[1, 3], aspect=1)
    ax4_t3 = plt.subplot(canv_grid[2, 3], aspect=1)
    ax4_t4 = plt.subplot(canv_grid[3, 3], aspect=1)

    ax5_t1 = plt.subplot(canv_grid[0, 4], aspect=1)
    ax5_t2 = plt.subplot(canv_grid[1, 4], aspect=1)
    ax5_t3 = plt.subplot(canv_grid[2, 4], aspect=1)
    ax5_t4 = plt.subplot(canv_grid[3, 4], aspect=1)

    ax_cbar = plt.subplot(canv[0, 1])


    #===============================================================================
    # make line plots

    x, y = np.meshgrid(range(9), range(9))

    for prop_class, ax_list in zip(
        prop_class_list,
        [
            [ax1_t1, ax1_t2, ax1_t3, ax1_t4],
            [ax2_t1, ax2_t2, ax2_t3, ax2_t4],
            [ax3_t1, ax3_t2, ax3_t3, ax3_t4],
            [ax4_t1, ax4_t2, ax4_t3, ax4_t4],
            [ax5_t1, ax5_t2, ax5_t3, ax5_t4],
        ]):
        for i, ax in enumerate(ax_list):
            time = time_list[i]
            # x, y, tb_num_corr = run.get_2body_corr_time(time)
            num_corr = prop_class.get_bjbibibj_time(time)

            # print(np.max(tb_num_corr))
            # print(np.min(tb_num_corr))
            im1 = ax.pcolormesh(x, y, np.transpose(num_corr), cmap='viridis',
                shading='flat', vmax=0.5)
            # im1 = ax.pcolormesh(x, y, np.transpose(tb_num_corr), cmap='coolwarm',
            #     shading='nearest', norm=divnorm)
            im1.set_rasterized(True)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
            ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
            ax.tick_params(axis="both", direction="in", color='orange')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')


    cbar1 = fig.colorbar(im1, cax=ax_cbar, extend='max')# , ticks=vticks)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.ax.set_ylabel(r'$\langle \hat{b}_j^\dagger \hat{b}_i^\dagger \hat{b}_i \hat{b}_j \rangle$', fontsize=16, labelpad=2)
    # cbar1.ax.yaxis.get_majorticklabels()[1].set_verticalalignment("top")




    ax1_t1.annotate(r'$t {=} 4$', fontsize=14, xy=(-0.45, .5), xycoords='axes fraction', color='black')
    ax1_t2.annotate(r'$t {=} 3$', fontsize=14, xy=(-0.45, .5), xycoords='axes fraction', color='black')
    ax1_t3.annotate(r'$t {=} 2$', fontsize=14, xy=(-0.45, .5), xycoords='axes fraction', color='black')
    ax1_t4.annotate(r'$t {=} 1$', fontsize=14, xy=(-0.45, .5), xycoords='axes fraction', color='black')



    ax1_t1.annotate(r'$\theta = 0$', fontsize=14, xy=(0.25, 1.07), xycoords='axes fraction', color='black')
    ax2_t1.annotate(r'$\theta = 0.2\,\pi$', fontsize=14, xy=(0.2, 1.07), xycoords='axes fraction', color='black')
    ax3_t1.annotate(r'$\theta = 0.5\,\pi$', fontsize=14, xy=(0.2, 1.07), xycoords='axes fraction', color='black')
    ax4_t1.annotate(r'$\theta = 0.8\,\pi$', fontsize=14, xy=(0.2, 1.07), xycoords='axes fraction', color='black')
    ax5_t1.annotate(r'$\theta = 1.0\,\pi$', fontsize=14, xy=(0.2, 1.07), xycoords='axes fraction', color='black')

    #===========================================================================
    path_fig.mkdir(parents=True, exist_ok=True)
    path_cwd = os.getcwd()
    os.chdir(path_fig)
    plt.savefig(fig_name)
    plt.close()

    if fig_name[-3:] == 'png':
        subprocess.check_output(["convert", fig_name, "-trim", fig_name])
    elif fig_name[-3:] == 'pdf':
        subprocess.check_output(["pdfcrop", fig_name, fig_name])

    os.chdir(path_cwd)
