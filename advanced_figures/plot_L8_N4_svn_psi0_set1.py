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
    N = 4
    L = 8
    U = 0.0

    theta_list = [el*np.pi for el in [0.0, 0.2]]
    nstate_list = [
        "4-0-0-0-0-0-0-0",
        "3-1-0-0-0-0-0-0",
        "0-1-1-1-1-0-0-0",
        "2-0-0-0-0-0-0-2",
        ]


    #---------------------------------------------------------------------------
    U_ = f'{U:.1f}'.replace('.', '_')
    fig_name = f'svn_psi0_set1_{bc_name}_L8_N4_U_{U_}.pdf'


    if bc_name == 'obc':
        bc = 'open'
    elif bc_name == 'pbc':
        bc = 'periodic'


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    fig.subplots_adjust(left=0.15, right=0.93, top=0.86, bottom=0.19)

    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------
    svn_lol = []
    for nstate_str in nstate_list:

        svn_th = []
        for theta in theta_list:
            print(nstate_str, f'thpi={theta/np.pi:.3f}')
            prop_class = Propagation(
                bc=bc,
                L=L,
                N=N,
                J=1,
                U=U,
                theta=theta,
                Tprop=6000,
                dtprop=1,
                psi0_str=f'psi0_nstate_{nstate_str}',
            )

            svn, svn_max_val = prop_class.nstate_SVN()
            time = prop_class.time

            ax.plot(time, svn)

    ax.axhline(svn_max_val, color='gray', ls='--')
    ax.axhline(np.log(330-80), color='black', ls='--')



    ax.set_ylabel(r'$S^{vN}$', fontsize=12, labelpad=2)

    ax.set_xlabel(r'$t \ [\hbar/J]$', fontsize=12)





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
