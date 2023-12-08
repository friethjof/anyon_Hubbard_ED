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


def make_plot(U, theta, bc_name):

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    theta_U_ = f'U_{U:.1f}_thpi_{theta/np.pi:.1f}'.replace('.', '_')
    fig_name = f'spectrum_degeneracy_{bc_name}_thpi_{theta_U_}.pdf'


    if bc_name == 'obc':
        bc = 'open'
    elif bc_name == 'pbc':
        bc = 'periodic'

    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    N_list = [1, 2, 3, 4, 5, 6, 7, 8]
    L_list = [1, 2, 3, 4, 5, 6, 7, 8]


    d_N4_L = []
    d_L8_N = []
    d_mat = np.zeros((8, 8))
    for i, N in enumerate(N_list):
        for j, L in enumerate(L_list):
            print(N, L)

            #---------------------------------------------------------------
            # solve hamilt
            hamilt_class = AnyonHubbardHamiltonian(
                bc=bc,
                L=L,
                N=N,
                J=1,
                U=U,
                theta=theta
            )

            E0_degenarcy = hamilt_class.get_E0_degeneracy()

            d_mat[i, j] = E0_degenarcy

            if N == 4:
                d_N4_L.append(E0_degenarcy)

            if L == 8:
                d_L8_N.append(E0_degenarcy)



    set_E0 = set(d_mat.flatten())
    set_E0 = [el for el in sorted(list(set_E0)) if not np.isnan(el)]
    col_list = ['royalblue', 'cornflowerblue', 'lightskyblue', 'turquoise',
                'limegreen', 'forestgreen', 'khaki',
                'gold', 'orange', 'tomato', 'firebrick']



    color_list = (
        matplotlib.cm.get_cmap("tab20").colors
        + matplotlib.cm.get_cmap("Dark2").colors
    )
    col_dict = {el:color_list[i] for i, el in enumerate(set_E0)}

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
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    # Plot our figure
    fig,ax = plt.subplots()
    # im = ax.imshow(d_mat, cmap=cm, norm=norm)
    x, y = np.meshgrid(range(1, 9), range(1, 9))
    im = plt.pcolormesh(x, y, d_mat, shading='nearest', cmap=cm, norm=norm)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cbar = fig.colorbar(im, format=fmt, ticks=tickz)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(r"$d(E=0)$",
        fontsize=14)

    ax.set_xlabel('$L$')
    ax.set_ylabel('$N$')





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
