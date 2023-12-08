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


def make_plot():

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    fig_name = f'E0_degeneracy_U0.pdf'
    U = 0
    bc = 'open'

    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------
    N_list = [1, 2, 3, 4, 5, 6, 7, 8]
    L_list = [1, 2, 3, 4, 5, 6, 7, 8]

    d_mat_th_0 = np.zeros((8, 8))
    d_mat_th_finite = np.zeros((8, 8))
    for i, N in enumerate(N_list):
        for j, L in enumerate(L_list):
            print(N, L)

            #---------------------------------------------------------------
            hamilt_class = AnyonHubbardHamiltonian(
                bc=bc,
                L=L,
                N=N,
                J=1,
                U=U,
                theta=0
            )
            E0_degenarcy = hamilt_class.get_E0_degeneracy()
            d_mat_th_0[i, j] = E0_degenarcy

            #---------------------------------------------------------------
            hamilt_class = AnyonHubbardHamiltonian(
                bc=bc,
                L=L,
                N=N,
                J=1,
                U=U,
                theta=0.5*np.pi
            )
            E0_degenarcy = hamilt_class.get_E0_degeneracy()
            d_mat_th_finite[i, j] = E0_degenarcy


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3*0.65, 2.6), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.17)

    #split vertical
    canv = gridspec.GridSpec(1, 2, width_ratios=[1.0, 0.05], wspace=0.05)
    canv_grid = gridspec.GridSpecFromSubplotSpec(1, 2, canv[0, 0],
        width_ratios=[1, 1], wspace=0.05)

    ax1 = plt.subplot(canv_grid[0, 0], aspect=1)
    ax2 = plt.subplot(canv_grid[0, 1], aspect=1)
    axl = plt.subplot(canv[0, 1])



    #===========================================================================
    set_E0 = set(d_mat_th_0.flatten()).union(set(d_mat_th_finite.flatten()))
    set_E0 = [el for el in sorted(list(set_E0)) if not np.isnan(el)]
    col_list = ['royalblue', 'lightskyblue', 'turquoise',
                'mediumseagreen', 'yellowgreen',  'darkgreen', 'gold',
                'orange', 'tomato', 'indianred', 'darkred',
                'mediumpurple', 'indigo']

    col_dict = {el:col_list[i] for i, el in enumerate(set_E0)}

    # We create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

    # Let's also define the description of each category :
    # 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here !
    # Or using another dict maybe could help.
    labels = np.array(set_E0)
    len_lab = len(labels)

    # prepare normalizer
    ## Prepare bins for the normalizer
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

    ## Make normalizer and formatter
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: int(labels[norm(x)]))

    # Plot our figure
    # im = ax.imshow(d_mat, cmap=cm, norm=norm)
    x, y = np.meshgrid(range(1, 9), range(1, 9))
    im = ax1.pcolormesh(x, y, d_mat_th_0, shading='nearest', cmap=cm, norm=norm)
    ax2.pcolormesh(x, y, d_mat_th_finite, shading='nearest', cmap=cm, norm=norm)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cbar = fig.colorbar(im, cax=axl, format=fmt, ticks=tickz)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(r"$d(E=0)$",
        fontsize=14)
    cbar.ax.minorticks_off()

    ax1.set_xlabel('$L$')
    ax2.set_xlabel('$L$')
    ax1.set_ylabel('$N$')
    ax1.set_xticks([1,2,3,4,5,6,7,8])
    ax2.set_xticks([1,2,3,4,5,6,7,8])
    ax1.set_yticks([1,2,3,4,5,6,7,8])
    ax2.set_yticklabels([])

    for ax in [ax1, ax2]:
        [ax.axhline(i + 0.5, lw=0.5, c='lightgray') for i in range(1, 8)]
        [ax.axvline(i + 0.5, lw=0.5, c='lightgray') for i in range(1, 8)]


    ax1.set_title(r'$\theta=0$', fontsize=12)
    ax2.set_title(r'$\theta>0$', fontsize=12)

    ax1.annotate('(a)', fontsize=10, xy=(-0.05, 1.05), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=10, xy=(-0.05, 1.05), xycoords='axes fraction')


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
