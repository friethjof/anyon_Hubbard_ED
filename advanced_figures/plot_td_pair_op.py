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


def make_plot():

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    fix_U_or_theta = 'U'
    bc_name = 'obc'
    N = 2
    L = 8
    theta_list = [el*np.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]
    U = 100
    psi0_str = 'psi0_nstate_2-0-0-0-0-0-0-0'

    #---------------------------------------------------------------------------
    # fix_U_or_theta = 'theta'
    # bc_name = 'obc'
    # N = 2
    # L = 8
    # U_list = [0.0, 0.2, 2, 100]
    # psi0_str = 'psi0_nstate_2-0-0-0-0-0-0-0'


    #---------------------------------------------------------------------------
    if fix_U_or_theta == 'theta':
        theta_ = f'{theta/np.pi:.1f}'.replace('.', '_')
        par_list = U_list
        fig_name = f'pair_op_{bc_name}_L_{L}_N_{N}_thpi_{theta_}_{psi0_str}.pdf'
    elif fix_U_or_theta == 'U':
        U_ = f'{U:.1f}'.replace('.', '_')
        par_list = theta_list
        fig_name = f'pair_op_{bc_name}_L_{L}_N_{N}_U_{U_}_{psi0_str}.pdf'
    else:
        raise NotImplementedError

    if bc_name == 'obc':
        bc = 'open'
    elif bc_name == 'pbc':
        bc = 'periodic'

    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    #===========================================================================
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    fig.subplots_adjust(left=0.15, right=0.93, top=0.86, bottom=0.19)


    legend_list, line_list = [], []

    #---------------------------------------------------------------------------
    for i, par in enumerate(par_list):
        if fix_U_or_theta == 'theta':
            U = par
            if i == 0:
                legend_list.append(f'$U={par}$')
            else:
                legend_list.append(f'${par}$')

        elif fix_U_or_theta == 'U':
            theta = par
            if i == 0:
                legend_list.append(r'$\theta/\pi' + f'={par/np.pi:.1f}$')
            else:
                legend_list.append(f'${par/np.pi:.1f}$')

        # solve hamilt
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

        #-----------------------------------------------------------------------



        l1, = ax.plot(prop_class.time, prop_class.pair_operator())
        line_list.append(l1)


    ax.set_xlabel(r'$t \ [\hbar/J]$', fontsize=12)
    ax.set_ylabel(r'$\nu_p(t)$', fontsize=12)
    ax.set_xlim(0, 5)

    ax.legend(line_list, legend_list,
        bbox_to_anchor=(0.5, 1.1), ncol=5, loc='center', fontsize=10,
        borderpad=0.2, handlelength=1.5, handletextpad=0.6, labelspacing=0.2,
        columnspacing=1, framealpha=0.5)



    #===========================================================================
    path_fig.mkdir(parents=True, exist_ok=True)
    path_cwd = os.getcwd()
    os.chdir(path_fig)
    plt.savefig(fig_name)
    plt.close()

    # if fig_name[-3:] == 'png':
    #     subprocess.check_output(["convert", fig_name, "-trim", fig_name])
    # elif fig_name[-3:] == 'pdf':
    #     subprocess.check_output(["pdfcrop", fig_name, fig_name])

    os.chdir(path_cwd)
