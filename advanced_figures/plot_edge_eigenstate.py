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
from helper import other_tools


def make_plot(theta):

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    bc_name = 'obc'
    N = 4
    L = 12

    theta_ = f'{theta/np.pi:.1f}'.replace('.', '_')
    fig_name = f'edge_eigenstate_{bc_name}_L_{L}_N_{N}_thpi_{theta_}.pdf'


    if L == 8 and N == 4:
        ind_edge = 164
    elif L == 12 and N == 4:
        ind_edge = 686
    elif L == 8 and N == 6:
        ind_edge = 859

    if bc_name == 'obc':
        bc = 'open'
    elif bc_name == 'pbc':
        bc = 'periodic'



    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------
    U_list = [0.0, 0.2, 1.0, 2.0]

    edge_state_list = []
    ind_list = None
    ind_max = None
    for U in U_list:
        # solve hamilt
        hamilt_class = AnyonHubbardHamiltonian(
            bc=bc,
            L=L,
            N=N,
            J=1,
            U=U,
            theta=theta
        )
        print('U', U)
        E0_degenarcy = hamilt_class.get_E0_degeneracy()

        # get E0 degeneracies
        if U == 0:
            dict_degen = other_tools.find_degeneracies(hamilt_class.evals())
            E0_found = False
            for k, v in dict_degen.items():
                if abs(eval(k)) < 1e-10:
                    ind_list = v
                    E0_found = True
                    break
            assert E0_found

        if U > 0 and ind_max is None:
            natorb_E0ind_list = [
                hamilt_class.eigenstate_nOp(hamilt_class.evecs()[i])
                for i in ind_list
            ]
            # [plt.plot(orb) for orb in natorb_E0ind_list]
            # plt.show()
            # exit()
        edge_state = hamilt_class.eigenstate_nOp(hamilt_class.evecs()[ind_edge])
        edge_state_list.append(edge_state)
        # plt.plot(edge_state)
        # plt.show()
        # print(ind_list)
        # exit()



    fig, ax = plt.subplots()

    x_grid = range(1, L+1)

    for i, U in enumerate(U_list):
        ax.plot(x_grid, edge_state_list[i], label=f'$U={U}$')

    ax.set_xlabel('site i')
    ax.set_ylabel(r'$\langle \phi_i | \hat{n}_i | \phi_i \rangle $')

    ax.legend()



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
