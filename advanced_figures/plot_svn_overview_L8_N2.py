import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from multi_prop_class import MultiPropClass
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate import Propagation
import path_dirs


def make_plot():

    path_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm/exact_diagonalization/nice_figures')


    #---------------------------------------------------------------------------
    N = 2
    L = 8

    fig_name = f'svn_overview_obc_L8_N2.pdf'




    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.2*0.8, 8.2*0.5), dpi=300)
    fig.subplots_adjust(bottom=0.1, left=0.08, right=0.97, top=0.9)

    canv = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    canv_top = gridspec.GridSpecFromSubplotSpec(1, 2, canv[0, 0], width_ratios=[1, 1], wspace=0.2)
    canv_bot = gridspec.GridSpecFromSubplotSpec(1, 2, canv[1, 0], width_ratios=[1, 1], wspace=0.2)

    ax1 = plt.subplot(canv_top[0, 0])
    ax2 = plt.subplot(canv_top[0, 1])
    ax3 = plt.subplot(canv_bot[0, 0])
    ax4 = plt.subplot(canv_bot[0, 1])




    #---------------------------------------------------------------------------
    # time-dependent SVN
    #---------------------------------------------------------------------------
    nstate_str = "0-0-0-1-1-0-0-0"
    prop_class = Propagation(
        bc='open',
        L=L,
        N=N,
        J=1,
        U=0.0,
        theta=0.5*np.pi,
        Tprop=50,
        dtprop=0.1,
        psi0_str=f'psi0_nstate_{nstate_str}',
    )

    svn, svn_max_val = prop_class.nstate_SVN()
    svn_eigenstate = prop_class.nstate_eigenstate_SVN(nstate_str)


    ax1.plot(prop_class.time, svn, color='tomato')
    ax1.axhline(svn_eigenstate, color='cornflowerblue', ls='--')
    ax1.axhline(svn_max_val, color='gray', ls='--')


    ax1.set_ylabel(r'$S^{\mathrm{vN}}(t)$', fontsize=12, labelpad=2)
    ax1.set_xlabel(r'$t \ [\hbar/J]$', fontsize=12, labelpad=0)
    ax1.set_xlim(-0.4, 50)
    ax1.set_ylim(0, 3.7)
    ax1.set_title(r'$|\vec{n}\rangle=|0,0,0,1,1,0,0,0\rangle$', fontsize=10)

    #===========================================================================
    # sort number states for later
    #===========================================================================
    idx_11, idx_2 = [], []
    for j, bas in enumerate(prop_class.basis.basis_list):
        if 1 in bas:
            idx_11.append(j)
        else:
            idx_2.append(j)
    idx = np.array(idx_11 + idx_2)


    #===========================================================================
    # inset, participating eigenenergies
    #===========================================================================
    # get index of nstate_str in basis_list
    nstate_eval = np.array([eval(el) for el in nstate_str.split('-')])
    for i, nstate_i in enumerate(prop_class.basis.basis_list):
        if (nstate_eval == nstate_i).all():
            nstate_ind = i
            break

    # indices of eigenstates which have an overlap with nstate_str,
    # i.e., where the nstate_str-contribution to the eigenstate is non-zero
    eval_list, coeff_list = [], []
    for i, coeff in enumerate(prop_class.evecs()[:, nstate_ind]):
        eval_list.append(prop_class.evals()[i])
        coeff_list.append(np.abs(coeff)**2)
    i_range = range(1, len(eval_list)+1)



    ax_ins = ax1.inset_axes([0.28, 0.18, 0.5, 0.4])
    ax_ins.axhline(0, c='gray', ls='--', lw=1, zorder=0)
    # ax_ins.set_ylim(0.87, 1.0)
    ax_ins.set_xticks([1, 20, 36])
    ax_ins.tick_params(labelsize=8)
    # ax_ins.set_xlim(-2.5, 2.5)
    # ax_ins.set_yticks([-2, 0])

    # ax_ins.set_xlabel(r'$g_{AC}$', fontsize=8, labelpad=-8, x=0.7)
    ax_ins.set_xlabel(r'$i$', fontsize=10, labelpad=-5, x=0.7)
    # ax_ins.xaxis.get_majorticklabels()[1].set_horizontalalignment("right")
    ax_ins.set_ylabel(r"$\epsilon_i$", fontsize=10, labelpad=-2, y=0.55)



    im = ax_ins.scatter(i_range, eval_list, c=coeff_list, s=10, marker='o',
                        cmap='Blues', edgecolors='black', linewidths=0.1)

    divider = make_axes_locatable(ax_ins)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel(r'$|\langle \phi_i | \vec{n} \rangle|^2 $', fontsize=10)



    #===========================================================================
    # nstate, theta dependce
    #===========================================================================
    color_list_1 = ['tomato', 'cornflowerblue', 'forestgreen', 'yellow', 'mediumorchid']
    color_list_2 = ['darkred', 'midnightblue', 'darkgreen', 'gold', 'darkorchid']
    ms_list_1 = ['o', 'D', 'P', 'X', 'h']
    ms_list_2 = ['H', 'd', '*', 's', 'h']

    ls_list = []
    for i, U in enumerate([0, 10]):

        svn_fit_list, fit_err_list = [], []
        svn_eig_list = []

        theta_list = np.array([el*np.pi for el in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        for theta in theta_list:

            nstate_str = "0-0-0-1-1-0-0-0"
            prop_class = Propagation(
                bc='open',
                L=L,
                N=N,
                J=1,
                U=U,
                theta=theta,
                Tprop=1000,
                dtprop=0.1,
                psi0_str=f'psi0_nstate_{nstate_str}',
            )

            svn_fit, svn_err, svn_max_val =\
                prop_class.nstate_SVN_horizontal_fit(tmin=500, tmax=1000)

            svn_fit_list.append(svn_fit)
            fit_err_list.append(svn_err)


            svn_eig_list.append(prop_class.nstate_eigenstate_SVN(nstate_str))


        l1 = ax2.errorbar(theta_list/np.pi, svn_fit_list, yerr=fit_err_list,
                    c=color_list_1[i], marker=ms_list_1[i], ms=3, lw=1)
        ls_list.append(l1)

        l2, = ax2.plot(theta_list/np.pi, svn_eig_list, color=color_list_2[i],
                    marker=ms_list_2[i], ms=3, lw=1)
        ls_list.append(l2)


    ax2.legend(ls_list, [
        r'$\bar{S}^{\mathrm{vN}}$, $U{=}0$',
        r'$S_{\Phi_i}^{\mathrm{vN}}$, $U{=}0$',
        r'$\bar{S}^{\mathrm{vN}}$, $U{=}10$',
        r'$S_{\Phi_i}^{\mathrm{vN}}$, $U{=}10$'],
        bbox_to_anchor=(0.5, 1.15), ncol=2, loc='center', fontsize=8,
        framealpha=0.9, borderpad=0.2, handlelength=1.5, handletextpad=0.6,
        labelspacing=0.2, columnspacing=0.9, frameon=True)


    ax2.axhline(svn_max_val, color='gray', ls='--')


    ax2.set_xlabel(r'$\theta \, [\pi]$', fontsize=12, labelpad=0)
    ax2.set_ylim(2.3, 3.4)


    #===========================================================================
    # all nstate
    #===========================================================================
    color_list = ['tomato', 'cornflowerblue', 'forestgreen', 'yellow', 'mediumorchid']
    ms_list = ['o', 'D', 'P', 'X', 'h']
    theta_list = [el*np.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]

    multi_dict = {
        'bc':'periodic',
        'J':1,
        'L':8,
        'N':2,
        'Tprop':1000,
        'dtprop':0.1,
        'psi0s_str':'all_number_states'
    }




    multi_class = MultiPropClass(**multi_dict)
    theta_list = [el*np.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]

    #---------------------------------------------------------------------------
    psi0_list, obs_lol, err_lol = multi_class.prop_vary_U_or_theta(
        par2var='theta',
        U_list=[0],
        theta_list=theta_list,
        obs_name='SVN_fit',
        args_dict={'tmin':500, 'tmax':1000, 'return':True}
    )

    displ = 0
    for i, svn_list in enumerate(obs_lol):
        ax3.errorbar(np.array(i_range)+displ, svn_list[idx], yerr=err_lol[i][idx],
                    c=color_list[i], fmt=ms_list[i], ms=2, lw=1)
        displ += 0.1

    #---------------------------------------------------------------------------
    psi0_list, obs_lol, err_lol = multi_class.prop_vary_U_or_theta(
        par2var='theta',
        U_list=[10],
        theta_list=theta_list,
        obs_name='SVN_fit',
        args_dict={'tmin':500, 'tmax':1000, 'return':True}
    )

    displ = 0
    for i, svn_list in enumerate(obs_lol):
        ax4.errorbar(np.array(i_range)+displ, svn_list[idx], yerr=err_lol[i][idx],
                    c=color_list[i], fmt=ms_list[i], ms=2, lw=1)
        displ += 0.05



    #---------------------------------------------------------------------------
    for ax in [ax3, ax4]:

        xticks_pos = np.array(i_range)
        ax.set_xticks(i_range)
        ax.set_xticklabels([])
        ax.set_ylim(1.5, 3.7)
        ax.axhline(svn_max_val, color='gray', ls='--')



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
        y_half_brace = -(1/(1.+np.exp(-beta*(x_half-x_half[0])))
                        + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))

        y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
        y = brace_y + (.08*y - .01)*yax_span # adjust vertical position

        ax.autoscale(False)
        ax.plot(x, y, color='dimgray', lw=1, clip_on=False)

        ax.text(brace_xmin + brace_len/2, brace_y-0.5, text, ha='center',
            va='bottom', rotation=0, fontsize=10, color='dimgray')



    draw_brace(ax3, 1.5, (1, 28.8), r'$|\vec{n}_i\rangle=|\dots,1,\dots,1\dots\rangle$')
    draw_brace(ax3, 1.5, (29.2, 36), r'$|\dots,2,\dots\rangle$')

    draw_brace(ax4, 1.5, (1, 28.8), r'$|\dots,1,\dots,1\dots\rangle$')
    draw_brace(ax4, 1.5, (29.2, 36), r'$|\dots,2,\dots\rangle$')

    ax3.set_ylabel(r'$\bar{S}^{\mathrm{vN}}$', fontsize=12, labelpad=2)

    #===========================================================================
    # inset, energy
    #===========================================================================
    hamilt_class = AnyonHubbardHamiltonian(
        bc='open',
        L=L,
        N=N,
        J=1,
        U=10,
        theta=0.0,
    )


    ax_ins = ax4.inset_axes([0.09, 0.09, 0.4, 0.3])
    ax_ins.axhline(0, c='gray', ls='--', lw=1, zorder=0)
    ax_ins.set_xticks([1, 20, 36])
    ax_ins.tick_params(axis="x",direction="in", pad=2)
    ax_ins.tick_params(labelsize=6)

    ax_ins.set_xlabel(r'$i$', fontsize=9, labelpad=-7, x=0.7)
    ax_ins.set_ylabel(r"$\epsilon_i$", fontsize=10, labelpad=-5, y=0.55)
    ax_ins.patch.set_alpha(0.8)


    ax_ins.scatter(i_range, hamilt_class.evals(), s=5, marker='o', c='dimgray',)



    #===========================================================================
    ax1.annotate('(a)', fontsize=10, xy=(0.04, .7), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=10, xy=(0.02, .89), xycoords='axes fraction')
    ax3.annotate('(c)', fontsize=10, xy=(0.02, .84), xycoords='axes fraction')
    ax4.annotate('(d)', fontsize=10, xy=(0.02, .84), xycoords='axes fraction')



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


#===============================================================================
#===============================================================================
#===============================================================================
    # for i, theta in enumerate(theta_list):
    #     hamilt_class = AnyonHubbardHamiltonian(
    #         bc='open',
    #         L=L,
    #         N=N,
    #         J=1,
    #         U=1.0,
    #         theta=theta,
    #     )
    #     # sort:
    #     idx_11, idx_2 = [], []
    #     for j, bas in enumerate(hamilt_class.basis.basis_list):
    #         if 1 in bas:
    #             idx_11.append(j)
    #         else:
    #             idx_2.append(j)
    #     idx = np.array(idx_11 + idx_2)
    #
    #     nstate_eigenstate_SVN = hamilt_class.nstate_eigenstate_SVN()
    #     ax3.scatter(i_range, nstate_eigenstate_SVN[idx], s=7, color=color_list[i],
    #                 marker=ms_list[i])
    #
    #
    #
    # for i, theta in enumerate(theta_list):
    #     hamilt_class = AnyonHubbardHamiltonian(
    #         bc='open',
    #         L=L,
    #         N=N,
    #         J=1,
    #         U=100,
    #         theta=theta,
    #     )
    #
    #     # sort:
    #     idx_11, idx_2 = [], []
    #     for j, bas in enumerate(hamilt_class.basis.basis_list):
    #         if 1 in bas:
    #             idx_11.append(j)
    #         else:
    #             idx_2.append(j)
    #     idx = np.array(idx_11 + idx_2)
    #
    #     nstate_eigenstate_SVN = hamilt_class.nstate_eigenstate_SVN()
    #     ax4.scatter(i_range, nstate_eigenstate_SVN[idx], s=7, color=color_list[i],
    #                 marker=ms_list[i])
