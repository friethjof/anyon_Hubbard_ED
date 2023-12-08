import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate import Propagation
import path_dirs
from helper import other_tools


def make_plot():

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    bc_name = 'obc'
    fix_U_or_theta = 'U'
    N = 4
    L = 8
    theta = 0.2*np.pi
    U = 0.2
    psi0_str = 'psi0_n00n'
    vmax = 2

    #---------------------------------------------------------------------------
    theta_ = f'{theta/np.pi:.1f}'.replace('.', '_')
    U_ = f'{U:.1f}'.replace('.', '_')
    fig_name = f'num_op_diff_{bc_name}_L_{L}_N_{N}_{psi0_str}_U_{U_}_thpi_{theta_}.pdf'


    if bc_name == 'obc':
        bc = 'open'
    elif bc_name == 'pbc':
        bc = 'periodic'

    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    ham_dict = {
        'bc':bc,
        'L':L,
        'N':N,
        'J':1,
        'U':U,
        'theta':theta,
        'Tprop':5.0,
        'dtprop':0.1
    }

    prop_class = Propagation(
        **ham_dict,
        psi0_str=psi0_str,
    )

    time, num_op = prop_class.num_op_mat()

    #---------------------------------------------------------------------------
    ham_dict['N'] = 2
    prop_class = Propagation(
        **ham_dict,
        psi0_str='psi0_nstate_2-0-0-0-0-0-0-0'
    )
    time, num_op_left = prop_class.num_op_mat()

    prop_class = Propagation(
        **ham_dict,
        psi0_str='psi0_nstate_0-0-0-0-0-0-0-2'
    )
    time, num_op_right = prop_class.num_op_mat()

    x_grid = np.arange(1, L+1)

    x, t = np.meshgrid(x_grid, time)

    num_op_diff =  num_op - num_op_left - num_op_right


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3, 2.3), dpi=300)
    fig.subplots_adjust(left=0.06, right=0.91, top=0.9, bottom=0.25)

    #split vertical

    canv = gridspec.GridSpec(1, 2, width_ratios=[3.1, 1.05], wspace=0.2)

    grid_left = gridspec.GridSpecFromSubplotSpec(1, 4, canv[0, 0], wspace=0.12,
        width_ratios=[1, 1, 1, 0.05])

    grid_right = gridspec.GridSpecFromSubplotSpec(1, 2, canv[0, 1], wspace=0.1,
        width_ratios=[1, 0.05])



    ax1 = plt.subplot(grid_left[0, 0])
    ax2 = plt.subplot(grid_left[0, 1])

    ax3 = plt.subplot(grid_left[0, 2])
    ax4 = plt.subplot(grid_right[0, 0])

    axl1 = plt.subplot(grid_left[0, 3])
    axl2 = plt.subplot(grid_right[0, 1])

    #===========================================================================
    # make line plots
    for ax in [ax1, ax2, ax3, ax4]:
        # ax.set_ylim(0, 5)
        [ax.axvline(i+0.5, c='black', lw=2) for i in range(1, L)]
        ax.set_xticks(list(range(2, L+1, 2)))


    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])


    ax1.set_ylabel(r'$t \ \left[ \hbar/J \right]$', fontsize=12, labelpad=2)

    ax1.set_xlabel(r'site $i$', fontsize=12, labelpad=2)
    ax2.set_xlabel(r'site $i$', fontsize=12, labelpad=2)
    ax3.set_xlabel(r'site $i$', fontsize=12, labelpad=2)
    ax4.set_xlabel(r'site $i$', fontsize=12, labelpad=2)


    #---------------------------------------------------------------------------
    im1 = ax1.pcolormesh(x, t, np.transpose(num_op), cmap='turbo',
        shading='nearest', vmin=0.0, vmax=vmax)
    im1.set_rasterized(True)

    im2 = ax2.pcolormesh(x, t, np.transpose(num_op_left), cmap='turbo',
        shading='nearest', vmin=0.0, vmax=vmax)
    im2.set_rasterized(True)

    #-------------------------------------------------------------------------------
    im3 = ax3.pcolormesh(x, t, np.transpose(num_op_right), cmap='turbo',
        shading='nearest', vmin=0.0, vmax=vmax)
    im3.set_rasterized(True)

    im4 = ax4.pcolormesh(x, t, np.transpose(num_op_diff), cmap='bwr',
        shading='nearest', norm=colors.CenteredNorm())
    im4.set_rasterized(True)



    cbar1 = fig.colorbar(im1, cax=axl1)# , ticks=vticks)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.ax.set_ylabel(r'$\langle \hat{n}_i \rangle(t)$', fontsize=14, labelpad=5)

    # cbar1.ax.yaxis.get_majorticklabels()[1].set_verticalalignment("top")

    cbar2 = fig.colorbar(im4, cax=axl2)# , ticks=vticks)
    cbar2.ax.tick_params(labelsize=10)
    cbar2.ax.set_ylabel(r'$\Delta \hat{n}_i(t)$', fontsize=14, labelpad=5)


    ax1.annotate('(a)', fontsize=12, xy=(0.04, .87), xycoords='axes fraction', color='black',
        bbox=dict(facecolor='white', edgecolor='white',  alpha=0.7, linewidth=0, pad=1))
    ax2.annotate('(b)', fontsize=12, xy=(0.04, .87), xycoords='axes fraction', color='black',
        bbox=dict(facecolor='white', edgecolor='white',  alpha=0.7, linewidth=0, pad=1))
    ax3.annotate('(c)', fontsize=12, xy=(0.04, .87), xycoords='axes fraction', color='black',
        bbox=dict(facecolor='white', edgecolor='white',  alpha=0.7, linewidth=0, pad=1))
    ax4.annotate('(d)', fontsize=12, xy=(0.04, .87), xycoords='axes fraction', color='black',
        bbox=dict(facecolor='white', edgecolor='white',  alpha=0.7, linewidth=0, pad=1))


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
