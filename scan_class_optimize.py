import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from propagation.optimze import ClassOptimize
from propagation.propagate_td_ham import PropagationTD

import path_dirs
from helper import other_tools
from helper import plot_helper
from helper import operators


class ScanClassOptimize():
    """Organize exact diagonalization method, plotting, scaning of paramters.
    """

    def __init__(self, bc, J, L, N, U_ini, theta_ini, U_fin, theta_fin,
                 psi0_str, psifinal_str, dtprop=None):

        self.bc=bc
        self.L=L
        self.N=N
        self.J=J
        self.U_ini=U_ini
        self.theta_ini=theta_ini
        self.U_fin=U_fin
        self.theta_fin=theta_fin
        self.psi0_str=psi0_str
        self.psifinal_str=psifinal_str
        self.dtprop=dtprop


        self.path_fig_dir = path_dirs.get_path_fig_optimize(
            bc, L, N, U_ini, theta_ini, psi0_str, psifinal_str)


    #===========================================================================
    def optimize_vary_N_interp_Tprop(self, N_interp_list, Tprop_list,
                                     ini_guess_list, arg_dict):
        """ """

        for N_interp in N_interp_list:
            for Tprop in Tprop_list:
                for ini_guess in ini_guess_list:


                    #-----------------------------------------------------------
                    arg_dict['N_interp'] = N_interp
                    arg_dict['Tprop'] = Tprop
                    arg_dict['ini_guess'] = ini_guess

                    class_opt = ClassOptimize(
                        bc=self.bc,
                        L=self.L,
                        N=self.N,
                        J=self.J,
                        U_ini=self.U_ini,
                        theta_ini=self.theta_ini,
                        U_fin=self.U_fin,
                        theta_fin=self.theta_fin,
                        psi0_str=self.psi0_str,
                        psifinal_str=self.psifinal_str,
                        Tprop=Tprop,
                        dtprop=arg_dict['dtprop'],
                        version=arg_dict['version'],
                        N_interp=N_interp,
                        ini_guess=ini_guess
                    )

                    class_opt.perform_optimzation_bfgs()

                    dum_name = path_dirs.get_optimize_dum_name(
                        Tprop=Tprop,
                        version=arg_dict['version'],
                        N_interp=N_interp,
                        ini_guess=ini_guess
                    )


                    if True:
                        succ = class_opt.get_res(['success'])
                        time, U_list, theta_list, cost_list = class_opt.get_res(
                            ['time', 'U_list', 'theta_list', 'cost_list'])
                        print(time)
                        time_interp, U_interp, theta_interp = class_opt.get_res(
                            ['time_interp', 'U_interp', 'th_interp'])
                        # theta_interp = theta_interp%(2*np.pi)
                        # theta_list = theta_list%(2*np.pi)


                        plot_helper.plot_twinx(
                            fig_name=self.path_fig_dir/'optimized_th_U_cost'/(
                                     dum_name + '_res_cost.png'),
                            x_list=time,
                            y_lists1=[U_list, theta_list],
                            y_lists2=[cost_list],
                            label_list=[
                                r'$U(t)$',
                                r'$\theta(t)$',
                                r'$\mathrm{min}||Av - \Psi(t_{\mathrm{fin}})||$',
                            ],
                            style_list=[
                                {'color':'tomato'},
                                {'color':'cornflowerblue'},
                                {'color':'forestgreen'},
                            ],
                            fig_args={
                                'xlabel':'$t$',
                                'scatter1':np.array([time_interp, U_interp]),
                                'scatter2':np.array([time_interp, theta_interp]),
                                'scatter3':np.array([time[-1], cost_list[-1]]),
                                'ylabel1':r'$U(t)$, $\theta(t)$',
                                'ylabel2':r'$\mathrm{min}||Av - \Psi(t_{\mathrm{fin}})||$',
                                'ylim2':[-0.02, 1.1],
                                'hline1':0,
                                'hline2':1,
                                'anno':f'{cost_list[-1]:.3f}',
                                'anno2':f'succ: {succ[0]}',
                            }
                        )

                    # plt.plot(class_opt.get_res_name('time'), class_opt.get_res_name('U_list'))
                    # plt.plot(class_opt.get_res_name('time'), class_opt.get_res_name('theta_list'))
                    # plt.plot(class_opt.get_res_name('time'), class_opt.get_res_name('cost_list'))
                    # plt.show()
                    # exit()




                    # class_opt.get_res_name('time'), class_opt.get_res_name('U_list')


    #===========================================================================
    def probe_cost_landscape(self, Tprop, par2var, U_max_list, theta_max_list,
                             version, arg_dict=None):
        """Vary theta and U and plot the landscape of """


        if par2var == 'theta':
            par1_list, par2_list = U_max_list, theta_max_list
        elif par2var == 'U':
            par1_list, par2_list = theta_max_list, U_max_list
        else:
            raise NotImplementedError




        if version == 1:
            # path_fig_ = path_dirs.get_path_basis_optimize(
            #     self.bc, self.L, self.N, self.psi0s_str, self.Tprop, self.dtprop
            # )
            U_ini = arg_dict['U_ini']
            U_fin = arg_dict['U_fin']
            theta_ini = arg_dict['theta_ini']
            theta_fin = arg_dict['theta_fin']
            Tprop_ = f'Tprop_{Tprop:.1f}'.replace('.', '_')

            path_basis_dum = path_dirs.get_path_basis(
                self.bc, self.L, self.N, U_ini, theta_ini, self.J
                )/f'probe_cost_landscape_psi0_{self.psi0_str}'/(
                f'{Tprop_}_v{version}')

            path_fig_top = path_dirs.get_path_fig_top(self.bc, self.L, self.N)/(
                f'probe_cost_landscape_psi0_{self.psi0_str}')

            fig_name_dum = f'{Tprop_}_v{version}'


        #-----------------------------------------------------------------------
        for par1 in par1_list:

            cost_lol = []
            parfct_lol = []

            for par2 in par2_list:  # <-- par2var



                if par2var == 'theta':
                    U_center = 0
                    theta_center = par2

                elif par2var == 'U':
                    U_center = par2
                    theta_center = 0

                print(par2var, par2)

                #---------------------------------------------------------------
                if version == 1:
                    time_interp = np.linspace(0, Tprop, 3)
                    U_vals = [U_ini, U_center, U_fin]
                    theta_vals = [theta_ini, theta_center, theta_fin]

                    U_fct = InterpolatedUnivariateSpline(time_interp, U_vals, k=2)
                    theta_fct = InterpolatedUnivariateSpline(time_interp, theta_vals, k=2)

                    Uc_ = f'U_center_{U_center:.1f}'.replace('.', '_')
                    thc_ = f'theta_center_{theta_center:.1f}'.replace('.', '_')
                    Tprop_ = f'Tprop_{Tprop:.1f}'.replace('.', '_')
                    path_basis = path_basis_dum/f'{Uc_}_{thc_}_vary_{par2var}'
                    os.makedirs(path_basis, exist_ok=True)
                else:
                    raise NotImplementedError

                #---------------------------------------------------------------
                prop_class_TD = PropagationTD(
                    bc=self.bc,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=U_fct,
                    theta=theta_fct,
                    psi0_str=self.psi0_str,
                    Tprop=Tprop,
                    dtprop=self.dtprop,
                )
                time = prop_class_TD.time

                cost_list = prop_class_TD.get_E0_subpace_U0_theta0_cost_list(
                    path_basis=path_basis
                )

                cost_lol.append(cost_list)
                if par2var == 'U': parfct_lol.append(U_fct(time))
                if par2var == 'theta': parfct_lol.append(theta_fct(time))



            #===================================================================
            # plot

            if par2var == 'theta':
                plot_helper.make_cplot(
                    fig_name=path_fig_top/(fig_name_dum + '_var_theta_cost'),
                    xarr=time,
                    yarr=par2_list,
                    mat=cost_lol,
                    fig_args={
                        'xlabel':r'$t$',
                        'ylabel':r'$\theta_{center}$',
                        'clabel':'cost'
                    }
                )

                plot_helper.make_cplot(
                    fig_name=path_fig_top/(fig_name_dum + '_var_theta'),
                    xarr=time,
                    yarr=par2_list,
                    mat=parfct_lol,
                    fig_args={
                        'xlabel':r'$t$',
                        'ylabel':r'$\theta_{center}$',
                        'clabel':r'$\theta(t)$'
                    }
                )

            elif par2var == 'U':
                plot_helper.make_cplot(
                    fig_name=path_fig_top/(fig_name_dum + '_var_U_cost'),
                    xarr=time,
                    yarr=par2_list,
                    mat=cost_lol,
                    fig_args={
                        'xlabel':r'$t$',
                        'ylabel':r'$U_center$',
                        'clabel':'cost'
                    }
                )

                plot_helper.make_cplot(
                    fig_name=path_fig_top/(fig_name_dum + '_var_U'),
                    xarr=time,
                    yarr=par2_list,
                    mat=parfct_lol,
                    fig_args={
                        'xlabel':r'$t$',
                        'ylabel':r'$U_center$',
                        'clabel':r'$U(t)$'
                    }
                )


    def analyze_optimized_psi(self, obs_name, Tprop, N_interp, ini_guess,
                              version=1, dtprop=0.1):

        #-----------------------------------------------------------------------
        # load optimized data
        class_opt = ClassOptimize(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=self.J,
            U_ini=self.U_ini,
            theta_ini=self.theta_ini,
            U_fin=self.U_fin,
            theta_fin=self.theta_fin,
            psi0_str=self.psi0_str,
            psifinal_str=self.psifinal_str,
            Tprop=Tprop,
            dtprop=dtprop,
            version=version,
            N_interp=N_interp,
            ini_guess=ini_guess
        )
        time = class_opt.get_res_name('time')
        if time is None:
            print('File not found --> return None')
            return None
        psi_list = class_opt.get_res_name('psi_list')
        time_interp = class_opt.get_res_name('time_interp')
        U_interp = class_opt.get_res_name('U_interp')
        th_interp = class_opt.get_res_name('th_interp')
        U_fct = InterpolatedUnivariateSpline(time_interp, U_interp)
        theta_fct = InterpolatedUnivariateSpline(time_interp, th_interp)

        #-----------------------------------------------------------------------
        # create propagation class
        prop_class_TD = PropagationTD(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=self.J,
            U=U_fct,
            theta=theta_fct,
            psi0_str=self.psi0_str,
            Tprop=Tprop,
            dtprop=dtprop,
        )
        prop_class_TD._psi_t = psi_list


        #-----------------------------------------------------------------------
        # make path fig
        dum_name = path_dirs.get_optimize_dum_name(
            Tprop=Tprop,
            version=version,
            N_interp=N_interp,
            ini_guess=ini_guess
        )

        fig_name = self.path_fig_dir/obs_name/f'{dum_name}_{obs_name}.png'

        title = plot_helper.make_title(self.L, self.N, self.U_ini,
                                       self.theta_ini, psi0=self.psi0_str)
        print(fig_name)
        #-----------------------------------------------------------------------
        # make analysis
        if obs_name == 'nOp':
            time, num_op = prop_class_TD.num_op_mat()
            plot_helper.num_op_cplot(
                fig_name=fig_name,
                time=time,
                L=self.L,
                num_op_mat=num_op,
                title=title
            )
        
        if obs_name == 'checkerboard_2bd':
            check_list = prop_class_TD.checkerboard_2bd(bool_save=False)
            plot_helper.plot_lines(
                fig_name=fig_name,
                x_lists=[time],
                y_lists=[check_list],
                label_list=[''],
                fig_args={
                    'xlabel':r'$t$',
                    'ylabel':r'$\sum_{ij}(-1)^{i-j}(1-2\hat{n_i})(1-2\hat{n_j})$',
                    'title':title,
                }
            )
