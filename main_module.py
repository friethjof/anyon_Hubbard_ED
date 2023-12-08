import math

import numpy as np

from scan_class import ScanClass
from multi_prop_class import MultiPropClass

# author: Friethjof Theel
# date: 15.06.2022
# last modified: Dec. 2023



#===============================================================================
'scan_class gs'
#===============================================================================
############# open boundary conditions #########################################
if False:
    print('scan class ground state')

    par_dict = {
        'bc':'open',
        'J':1,
        'L':6,
        'N':2,
    }

    theta_list = [el*math.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]
    # theta_list = [0]
    U_list = [0, 1, 10]
    # U_list = [100]


    scan_class = ScanClass(**par_dict)
    #---------------------------------------------------------------------------
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigen_spectrum')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'K_operator')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_E0')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_E0_K_basis')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_E0_energy_basis')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'nstate_eigenstate_SVN')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_E0_nstate')

    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_num_op_2b')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_corr_2b')

    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_E0_perturbation',
    #     args_dict={'U_max':0.001, 'steps':5, 'order':1})

    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigenstate_E0_perturbation',
    #     args_dict={'U_max':0.001, 'steps':5, 'order':2})

    # scan_class.plot_gs_multi_lines(
    #     U_list=U_list,
    #     theta_list=theta_list,
    #     obs_name='nstate_eigenstate_SVN'
    # )

    # scan_class.plot_gs_multi_lines(
    #     U_list=U_list,
    #     theta_list=theta_list,
    #     obs_name='bipartite_SVN'
    # )

    # scan_class.plot_gs_multi_lines(
    #     U_list=[0, 1, 10],
    #     theta_list=theta_list,
    #     obs_name='evec_2b_checkerboard'
    # )

    #---------------------------------------------------------------------------
    # theta-dependent eigenspectrum + zoom in
    #---------------------------------------------------------------------------
    # dth = 0.001*np.pi
    # scan_class.plot_gs_multi_lines(
    #     U_list=[0],
    #     theta_list=np.arange(0*np.pi, np.pi+dth, dth),
    #     obs_name='spectrum',
    #     args_dict={'dth':dth, 'n_evals':15}
    # )
    # dth = 0.001*np.pi
    # scan_class.plot_gs_multi_lines(
    #     U_list=[0],
    #     theta_list=np.arange(0.68*np.pi, 0.77*np.pi, dth),
    #     obs_name='spectrum',
    #     args_dict={
    #         'dth':dth,
    #         'n_evals':15,
    #         'ylim':[-2.1, -1.85],
    #     }
    # )
    # dth = 0.001*np.pi
    # scan_class.plot_gs_multi_lines(
    #     U_list=[0],
    #     theta_list=np.arange(0.8*np.pi, 0.9*np.pi, dth),
    #     obs_name='spectrum',
    #     args_dict={
    #         'dth':dth,
    #         'n_evals':15,
    #         'ylim':[-1.5, -1.3],
    #     }
    # )


############# periodic boundary conditions #####################################
if False:
    print('scan class ground state')

    par_dict = {
        'bc':'periodic',
        'J':1,
        'L':8,
        'N':4,
    }

    theta_list = [el*math.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]
    U_list = [0, 1, 10, 100]


    scan_class = ScanClass(**par_dict)
    #---------------------------------------------------------------------------
    scan_class.gs_vary_U_theta(U_list, theta_list, 'eigen_spectrum')




#===============================================================================
'scan_class dynamic'
#===============================================================================
############# open boundary conditions #########################################
if True:
    print('scan class propagation')

    par_dict = {
        'bc':'open',
        'J':1,
        'L':6,
        'N':2,
        'Tprop':20,
        'dtprop':0.1
    }

    theta_list = [el*math.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]
    U_list = [0, 1, 100]


    #---------------------------------------------------------------------------
    par_dict['psi0_str'] = 'psi0_n00n'
    # par_dict['psi0_str'] = 'pi0_eigstate_160'
    # par_dict['psi0_str'] = 'psi0_nstate_0-0-0-4'
    # par_dict['psi0_str'] = 'psi0_nstate_0-1-1-1-1-0-0-0'
    # par_dict['psi0_str'] = 'psi0_nstate_0-3-1-0-0-0-0-0'
    # par_dict['psi0_str'] = 'psi0_nstate_2-0-0-0-0-0-0-2'
    # par_dict['psi0_str'] = 'psi0_nstate_1-0-1-0-0-0-0-0'
    # par_dict['psi0_str'] = 'psi0_nstate_1-0-0-1-1-0-0-0'


    scan_class = ScanClass(**par_dict)
    #---------------------------------------------------------------------------
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op_2b',
    #                              arg_dict={'t_list':[0, 1, 2, 3, 5]})
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'K_operator')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'momentum_distribution_cont')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'momentum_distribution_discrete')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'nstate')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'nstate_SVN')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'natpop')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'eigenstate_projection')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'pair_operator')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'checkerboard_2bd')
    print('here')
    scan_class.prop_vary_U_theta(U_list, theta_list, 'E0_subspace_overlap')



#===============================================================================
'multi class dynamic'
#===============================================================================
if False:
    print('multi class propagation')

    par_dict = {
        'bc':'periodic',
        'J':1,
        'L':8,
        'N':2,
        'Tprop':500,
        'dtprop':0.1
    }

    theta_list = [el*math.pi for el in [0.0, 0.2, 0.5, 0.8, 1.0]]
    U_list = [0, 1, 5, 10, 100]
    # theta_list = [el*math.pi for el in [0.0]]
    # U_list = [0, 1, 5, 10, 100]

    #---------------------------------------------------------------------------
    par_dict['psi0s_str'] = 'all_number_states'
    # par_dict['psi0s_str'] = 'psi0_nstate_2-0-0-0-0-0-0-0'


    multi_class = MultiPropClass(**par_dict)
    #---------------------------------------------------------------------------
    # multi_class.prop_vary_U_theta(U_list, theta_list, 'pair_fit')
    # multi_class.prop_vary_U_or_theta('U', U_list, theta_list, 'pair_fit')
    # multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'pair_fit')

    # multi_class.prop_vary_U_theta(U_list, theta_list, 'SVN_fit')
    # multi_class.prop_vary_U_or_theta('U', U_list, theta_list, 'SVN_fit')
    # multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'SVN_fit', args_dict={'tmin':5000, 'tmax':6000})
    multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'SVN_fit', args_dict={'tmin':400, 'tmax':500})
    multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'SVN_max', args_dict={'tmin':400, 'tmax':500})


    # multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'SVN_max', args_dict={'tmin':1000, 'tmax':6000})
    # multi_class.prop_vary_U_theta(U_list, theta_list, 'SVN_max')
    # multi_class.prop_vary_U_or_theta('U', U_list, theta_list, 'SVN_max')
    # multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'SVN_max')

    # multi_class.prop_vary_U_theta(U_list, theta_list, 'natpop_SVN_max')
    # multi_class.prop_vary_U_or_theta('U', U_list, theta_list, 'natpop_SVN_max')
    # multi_class.prop_vary_U_or_theta('theta', U_list, theta_list, 'natpop_SVN_max')



#===============================================================================
'Kwan - paper'
#===============================================================================
if False:
    print('scan class propagation  -- Kwan')

    par_dict = {
        'bc':'open',
        'J':1,
        'L':20,
        'N':2,
        'Tprop':5,
        'dtprop':0.1
    }

    theta_list = [el*math.pi for el in [0.0, 0.5, 1.0]]
    U_list = [0.0]


    #---------------------------------------------------------------------------
    par_dict['psi0_str'] = 'psi0_0nn0'


    scan_class = ScanClass(**par_dict)
    #---------------------------------------------------------------------------
    # gs
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigen_spectrum')
    # scan_class.gs_vary_U_theta(U_list, theta_list, 'eigen_spectrum_mom')

    # dynamic
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op_bound_state')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op_bound_state_RMS')
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op_2b',
    #                              arg_dict={'t_list':[2.4]})
    # scan_class.prop_vary_U_theta(U_list, theta_list, 'op_2b_corr',
    #                              arg_dict={'t_list':[2.4]})



    scan_class.plot_multi_lines(
        U_list=U_list,
        obs_name='bound_scatter_root_mean_square_slope'
    )
    scan_class.plot_multi_lines(
        U_list=U_list,
        obs_name='bound_scatter_root_mean_square_slope_ratio'
    )
