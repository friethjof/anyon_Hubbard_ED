import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.colors import ListedColormap
from scipy import linalg

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian


def make_plot(L, N):

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    fig_name = f'compare_E0_subspace_L{L}_N{N}.pdf'
    U = 0
    bc = 'open'


    theta_list = [el*np.pi for el in np.arange(0, 1.01, 0.1)]
    # print(theta_list)

    hamilt_class = AnyonHubbardHamiltonian(
        bc=bc,
        L=L,
        N=N,
        J=1,
        U=U,
        theta=0.0*np.pi
    )
    evecs_E0_th0 = hamilt_class.get_eigenstates_E0()
    print(evecs_E0_th0)

    # vec_test = 0.5*evecs_E0_th0[0] + evecs_E0_th0[1] - 0.5*evecs_E0_th0[2]
    # vec_test = np.linalg.norm(vec_test)*vec_test
    #
    # x = np.linalg.lstsq(evecs_E0_th0.T, vec_test, rcond=None)[0]
    # zero_vec = np.abs((evecs_E0_th0.T).dot(x) - vec_test)
    # print(np.abs(np.vdot(zero_vec, zero_vec))**2)
    # exit()

    ovlp_list = []
    for th in theta_list:
        print(f'theta : {th/np.pi:.1f}')
        hamilt_class = AnyonHubbardHamiltonian(
            bc=bc,
            L=L,
            N=N,
            J=1,
            U=U,
            theta=th
        )

        evecs_E0 = hamilt_class.get_eigenstates_E0()

        sum_ovlp = 0
        for ev0 in evecs_E0_th0:
            x = np.linalg.lstsq(evecs_E0.T, ev0, rcond=None)[0]
            zero_vec = np.abs((evecs_E0.T).dot(x) - ev0)
            length_vec = np.abs(np.vdot(zero_vec, zero_vec))**2
            sum_ovlp += length_vec

        ovlp_list.append(sum_ovlp)


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3*0.65, 3), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.17)

    #split vertical
    canv = gridspec.GridSpec(2, 1, height_ratios=[0.05, 1.0], hspace=0.05)
    # canv_grid = gridspec.GridSpecFromSubplotSpec(1, 2, canv[0, 0],
    #     width_ratios=[1, 1], wspace=0.05)

    ax1 = plt.subplot(canv[1, 0])
    axl = plt.subplot(canv[0, 0])
    axl.set_axis_off()


    #===========================================================================

    ax1.scatter([el/np.pi for el in theta_list], ovlp_list)


    ax1.set_xlabel(r'$\theta$ [$\pi$]', fontsize=12)




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
