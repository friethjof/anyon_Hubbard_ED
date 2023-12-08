import numpy as np

# author: Friethjof Theel
# date: 15.06.2022
# last modified: Sept. 2023


from scan_class_td_ham import ScanClassTD
from propagation.optimze import ClassOptimize
from scan_class_optimize import ScanClassOptimize
from helper import plot_helper


#===============================================================================
'scan_class'
#===============================================================================
############# open boundary conditions #########################################
if False:
    par_dict = {
        'bc':'open',
        'J':1,
        'L':6,
        'N':2,
        'Tprop':20,
        'dtprop':0.1
    }

    theta_list = [el*np.pi for el in [0.0]]
    U_list = [
        lambda t: np.sin(t)
    ]

    #---------------------------------------------------------------------------
    par_dict['psi0_str'] = 'psi0_nstate_0-0-1-1-0-0'


    scan_class = ScanClassTD(**par_dict)

    #---------------------------------------------------------------------------
    scan_class.prop_vary_U_theta(U_list, theta_list, 'num_op')




############# optimize U(t), theta(t) ---- probe landscape #####################
if False:
    par_dict = {
        'bc':'open',
        'J':1,
        'L':6,
        'N':2,
        'U_ini':0,
        'theta_ini':0,
        'U_fin':0,
        'theta_fin':0,
        'dtprop':0.1
    }


    #---------------------------------------------------------------------------
    par_dict['psi0_str'] = 'psi0_nstate_0-0-1-1-0-0'
    par_dict['psifinal_str'] = 'E0_subpace_U0_theta0'


    class_opt = ScanClassOptimize(**par_dict)


    U_max_list = np.linspace(-5, 5, 41)
    theta_max_list = np.linspace(-5, 5, 41)


    for Tprop in [5, 7, 10, 15]:
        class_opt.probe_cost_landscape(
            Tprop=Tprop,
            par2var='theta',
            U_max_list=[0],
            theta_max_list=theta_max_list,
            version=1,
            arg_dict={'U_ini':0, 'theta_ini':0, 'U_fin':0, 'theta_fin':0}
        )

        class_opt.probe_cost_landscape(
            Tprop=Tprop,
            par2var='U',
            U_max_list=theta_max_list,
            theta_max_list=[0],
            version=1,
            arg_dict={'U_ini':0, 'theta_ini':0, 'U_fin':0, 'theta_fin':0}
        )

    exit()

############# get optimized U(t), theta(t) #####################################
if False:
    par_dict = {
        'bc':'open',
        'J':1,
        'L':6,
        'N':2,
        'U_ini':0,
        'theta_ini':0,
        'U_fin':0,
        'theta_fin':0,
    }

    # N_interp_list = [2, 3, 4, 5, 7]
    # Tprop_list = [2, 5, 7, 10]
    # ini_guess_list = [0.01, 0.1, 0.2, 1.0]

    N_interp_list = [2, 3, 4, 5, 7, 10]
    Tprop_list = [15, 20]
    ini_guess_list = [0.01, 0.1, 0.2, 1.0]

    # N_interp_list = [2]
    # Tprop_list = [1]
    # ini_guess_list = [0]

    #---------------------------------------------------------------------------
    # par_dict['psi0_str'] = 'psi0_nstate_0-0-1-1-0-0'
    par_dict['psi0_str'] = 'eigstate_0'
    par_dict['psifinal_str'] = 'E0_subpace_U0_theta0'



    class_opt = ScanClassOptimize(**par_dict)

    class_opt.optimize_vary_N_interp_Tprop(
        N_interp_list =N_interp_list,
        Tprop_list    =Tprop_list,
        ini_guess_list=ini_guess_list,
        arg_dict={
            'version':1,
            'dtprop':0.1,
        }
    )



############# optimize U(t), theta(t) --- analyze optimzed result ##############
if True:
    par_dict = {
        'bc':'open',
        'J':1,
        'L':6,
        'N':2,
        'U_ini':0,
        'theta_ini':0,
        'U_fin':0,
        'theta_fin':0,
    }
    par_dict['psi0_str'] = 'psi0_nstate_0-0-1-1-0-0'
    par_dict['psifinal_str'] = 'E0_subpace_U0_theta0'

    class_opt = ScanClassOptimize(**par_dict)

    #---------------------------------------------------------------------------
    # class_opt.analyze_optimized_psi(
    #     obs_name='nOp',
    #     Tprop=20,
    #     N_interp=10,
    #     ini_guess=1,
    # )

    # class_opt.analyze_optimized_psi(
    #     obs_name='checkerboard_2bd',
    #     Tprop=20,
    #     N_interp=10,
    #     ini_guess=1,
    # )
