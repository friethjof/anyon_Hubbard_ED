import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian


def make_plot():

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    L = 6
    N = 2

    assert L == 6 and N == 2


    # theta = 0.8*np.pi
    # U = 0.0
    # U_ = f'{U:.1f}'.replace('.', '_')
    # thpi_ = f'{theta/np.pi:.1f}'.replace('.', '_')
    # fig_name = f'nstate_ovlp_E0_subspace_L{L}_N{N}_U_{U_}_thpi_{thpi_}.pdf'

    U = 0.0
    U_ = f'{U:.1f}'.replace('.', '_')
    fig_name = f'energyOccupNstate_ovlp_E0_subspace_L{L}_N{N}_U_{U_}.pdf'


    leg_lab = []
    ovlp_lol = []
    for theta in [0.0, 0.2, 0.5, 0.8, 1.0]:

        hamilt_class = AnyonHubbardHamiltonian(
            bc='open',
            L=L,
            N=N,
            J=1,
            U=U,
            theta=theta*np.pi,
        )

        ovlp_lol.append(hamilt_class.ovlp_energy_occup_basis_E0_subspace())


    i_range = list(range(hamilt_class.basis.length))

    xticks_lab = []
    idx = np.argsort(ovlp_lol[0])
    for i in idx:
        nstate = hamilt_class.basis.basis_list[i]
        xticks_lab.append('-'.join([str(el) for el in nstate]))

    ovlp_lol = [el[idx] for el in ovlp_lol]


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.2*0.6, 8.2*0.5), dpi=300)
    fig.subplots_adjust(bottom=0.15, left=0.12, right=0.97, top=0.9)

    canv = gridspec.GridSpec(2, 1, height_ratios=[0.01, 1], hspace=0.1)

    ax1 = plt.subplot(canv[1, 0])

    axl = plt.subplot(canv[0, 0])
    axl.set_axis_off()





    #===========================================================================
    style_list = [
        {'marker':'s', 'color':'tomato'},
        {'marker':'o', 'color':'cornflowerblue'},
        {'marker':'d', 'color':'orange'},
        {'marker':'H', 'color':'forestgreen'},
        {'marker':'v', 'color':'gold'},
    ]
    l_list = []
    for i, ovlp in enumerate(ovlp_lol):
        l_ = ax1.scatter(i_range, ovlp, s=30, **style_list[i])
        l_list.append(l_)

    # ax1.set_xlabel(r'$\bar{n}_i$', fontsize=12, labelpad=2)
    ax1.set_ylabel(r'$\langle \{ \tilde{\eta}_1\cdots \tilde{\eta}_6\}| \Phi_{E=0} \rangle$', fontsize=12, labelpad=5)

    ax1.set_xticks(i_range)
    ax1.set_xticklabels(xticks_lab, rotation=50, ha='right', fontsize=8,
                   rotation_mode='anchor')
    ax1.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)

    ax1.legend(l_list,
        [r'$\theta=0$', r'$0.2\pi$', r'$0.5\pi$', r'$0.8\pi$', r'$\pi$'],
        bbox_to_anchor=(0.5, 1.1), ncol=5, loc='center', fontsize=12,
        borderpad=0.2, handlelength=1.5, handletextpad=0.6, labelspacing=0.2,
        columnspacing=1, framealpha=0.5)



    #===========================================================================
    def draw_brace(ax, brace_y, brace_xmin_xmax, text):
        """Draws an annotated brace on the axes."""
        brace_xmin, brace_xmax = brace_xmin_xmax
        brace_len = brace_xmax - brace_xmin

        ax_xmin, ax_xmax = ax.get_xlim()
        xax_span = ax_xmax - ax_xmin

        ax_ymin, ax_ymax = ax.get_ylim()
        yax_span = ax_ymax - ax_ymin

        resolution = int(brace_len/xax_span*50)*2+1 # guaranteed uneven
        beta = 200./brace_len # the higher this is, the smaller the radius

        # print(ymin, ymax)
        x = np.linspace(brace_xmin, brace_xmax, resolution)
        x_half = x[:int(resolution/2)+1]
        y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                        + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))

        y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
        y = brace_y + (.08*y - .01)*yax_span # adjust vertical position

        ax.autoscale(False)
        print(y)
        ax.plot(x, y, color='dimgray', lw=1, clip_on=False)

        # ax.text(brace_xmin + brace_len/2, brace_y+0.04, text, ha='center',
        #     va='bottom', rotation=0, fontsize=10, color='dimgray')


    # draw_brace(ax1, 0.3, (9, 14), r'$|\vec{n}_i\rangle=|\dots,2,\dots\rangle$')
    # draw_brace(ax1, -0.5, (29.2, 36), r'$|\dots,2,\dots\rangle$')








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
