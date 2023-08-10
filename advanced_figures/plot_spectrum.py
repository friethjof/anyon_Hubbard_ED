import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
import path_dirs
from helper import other_tools


def make_plot(L, N, U):

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    U_ = f'{U:.1f}'.replace('.', '_')
    fig_name = f'L{L}_N{N}_U_{U_}_energy_spectrum.pdf'

    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    theta_list = [el*np.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]

    eval_theta = []
    degeneracies_theta = []
    for theta in theta_list:
        #---------------------------------------------------------------
        # solve hamilt
        path_basis = other_tools.get_path_basis(
            path_dirs.path_data_top, L, N, U, theta)

        hamilt_class = AnyonHubbardHamiltonian(
            path_basis=path_basis,
            L=L,
            N=N,
            J=1,
            U=U,
            theta=theta
        )

        N_nstates = hamilt_class.basis.length

        evals = hamilt_class.energy_spectrum()
        eval_theta.append(evals)

        dict_degen = hamilt_class.energy_degeneracy()
        degeneracies_theta.append(dict_degen)

        # for key, val in dict_degen.items():
        #     print(key, val)


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3, 2.5), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.8, bottom=0.15)

    #split vertical
    canv = gridspec.GridSpec(2, 1, height_ratios=[0.08, 1.0], hspace=0.15)

    canv_grid = gridspec.GridSpecFromSubplotSpec(1, 5, canv[1, 0],
        width_ratios=[1, 1, 1, 1, 1], wspace=0.15)

    ax1 = plt.subplot(canv_grid[0, 0])
    ax2 = plt.subplot(canv_grid[0, 1])
    ax3 = plt.subplot(canv_grid[0, 2])
    ax4 = plt.subplot(canv_grid[0, 3])
    ax5 = plt.subplot(canv_grid[0, 4])

    axl = plt.subplot(canv[0, 0])
    axl.set_axis_off()

    #===========================================================================
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        # ax.set_xlim(0, N_nstates)
        ax.set_ylim(-8, 8)
        ax.set_xlabel('$i$')
    ax1.set_ylabel(r'$E_i$', fontsize=12)

    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])
    ax5.set_yticklabels([])
    # ax2.set_yticks([0, 5, 10, 15, 20, 25])
    # ax2.set_xlabel(r'$|\vec{n}_i\rangle$', fontsize=12)
    # ax2.set_ylabel(r'$t \ \left[ \hbar/J \right]$', fontsize=12)

    #---------------------------------------------------------------------------
    # s = 0.3
    # sd = 0.7
    s = 7
    sd = 10


    x_data = range(N_nstates)

    ax1.scatter(x_data, eval_theta[0], s=s, fc='black', marker='o')
    ax2.scatter(x_data, eval_theta[1], s=s, fc='black', marker='o')
    ax3.scatter(x_data, eval_theta[2], s=s, fc='black', marker='o')
    ax4.scatter(x_data, eval_theta[3], s=s, fc='black', marker='o')
    ax5.scatter(x_data, eval_theta[4], s=s, fc='black', marker='o')


    degen_colors = {
        '2': 'tomato',
        '4': 'cornflowerblue',
        '5': 'forestgreen',
        '10': 'gold',
        }
    # highlight degenarcies
    degen_used_colors = []
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
        dict_degen = degeneracies_theta[i]

        for e_str, ind_list in dict_degen.items():

            if len(ind_list) == 1:
                continue

            d = len(ind_list)
            for ind in ind_list:
                degen_used_colors.append(d)
                ax.scatter(ind, eval(e_str), s=sd, color=degen_colors[str(d)])



    degen_used_colors = set(degen_used_colors)
    if len(degen_used_colors) == 1:
        m2 = plt.scatter([], [], marker='o', color=degen_colors['2'], s=5)
        axl.legend([m2], [r'$\mathrm{degeneracy}=2$'],
            bbox_to_anchor=(0.5, 2.5), ncol=4, loc='center', fontsize=12,
            borderpad=0.2, handlelength=1.5, handletextpad=0.6, labelspacing=0.2,
            columnspacing=1, framealpha=0.5)


    elif len(degen_used_colors) == 4:
        m2 = plt.scatter([], [], marker='o', color=degen_colors['2'], s=5)
        m4 = plt.scatter([], [], marker='o', color=degen_colors['4'], s=5)
        m5 = plt.scatter([], [], marker='o', color=degen_colors['5'], s=5)
        m10 = plt.scatter([], [], marker='o', color=degen_colors['10'], s=5)
        axl.legend([m2, m4, m5, m10], [r'$\mathrm{degeneracy}=2$', r'$4$',
            r'$5$', r'$10$'
            ], bbox_to_anchor=(0.5, 2.5), ncol=4, loc='center', fontsize=12,
            borderpad=0.2, handlelength=1.5, handletextpad=0.6, labelspacing=0.2,
            columnspacing=1, framealpha=0.5)


    #===========================================================================
    ax1.annotate(r'$\theta=0$', fontsize=12, xy=(0.35, 1.04), xycoords='axes fraction')
    ax2.annotate(r'$\theta=0.2\pi$', fontsize=12, xy=(0.3, 1.04), xycoords='axes fraction')
    ax3.annotate(r'$\theta=0.5\pi$', fontsize=12, xy=(0.3, 1.04), xycoords='axes fraction')
    ax4.annotate(r'$\theta=0.8\pi$', fontsize=12, xy=(0.3, 1.04), xycoords='axes fraction')
    ax5.annotate(r'$\theta=\pi$', fontsize=12, xy=(0.35, 1.04), xycoords='axes fraction')



    ax1.annotate('(a)', fontsize=12, xy=(0.04, .91), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=12, xy=(0.04, .91), xycoords='axes fraction')
    ax3.annotate('(c)', fontsize=12, xy=(0.04, .91), xycoords='axes fraction')
    ax4.annotate('(d)', fontsize=12, xy=(0.04, .91), xycoords='axes fraction')
    ax5.annotate('(e)', fontsize=12, xy=(0.04, .91), xycoords='axes fraction')



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
