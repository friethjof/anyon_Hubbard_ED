import os
import math
from pathlib import Path
import decimal



def get_path_names(type, bc):
    if type == 'basis':
        path_basis = Path(f'/afs/physnet.uni-hamburg.de/project/zoq_t/harm_osc_coll/12/project_ahm_ED')
        if bc == 'open':
            return path_basis/'open_boundary_conditions'
        elif bc == 'periodic':
            return path_basis/'periodic_boundary_conditions'
        else:
            raise NotImplementedError

    elif type == 'fig':
        path_proj = Path(f'/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/Schreibtisch/project_ahm/exact_diagonalization')
        if bc == 'open':
            return path_proj/'figures_obc'
        elif bc == 'periodic':
            return path_proj/'figures_pbc'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_path_basis(bc, L, N, U, theta, J):
    """Create path of data from input pars. J is assumed to be always 1

    Args:
        path_data_top (Path): top path of data folder
        L (int): number of lattice sites
        N (int): number of atoms
        U (float): on-site interaction
        theta (float): statistical angle

    Returns:
        path_basis (Path): path where solved hamilt spectrum is stored
    """


    # count trailing zeros:

    assert isinstance(bc, str)

    th_exp = decimal.Decimal(str(round(theta/math.pi, 8))).as_tuple().exponent
    if th_exp >= -3:
        theta_ = f'{theta/math.pi:.3f}'.replace('.', '_')
    else:
        theta_ = f'{round(theta/math.pi,8)}'.replace('.', '_')



    if J != 1:
        if J == 0 and U == 1:
            U_ = 'inf'
        else:
            err_msg = f'U={U}, J={J}; J should be 0 and U=1 for U=inf!'
            raise ValueError(err_msg)
    else:
        U_exp = decimal.Decimal(str(round(theta/math.pi, 8))).as_tuple().exponent
        if th_exp >= -3:
            U_ = f'{U:.3f}'.replace('.', '_')
        else:
            U_ = f'{round(U, 8)}'.replace('.', '_')



    path_basis = get_path_names('basis', bc)/f'L{L}_N{N}/U_{U_}_thpi_{theta_}'
    path_basis.mkdir(parents=True, exist_ok=True)

    return path_basis


def get_path_fig_top(bc, L, N):
    return get_path_names('fig', bc)/f'L{L}_N{N}'


def get_path_fig_top_dyn(bc, L, N, psi0_str, Tprop, dtprop):
    path_fig_top = get_path_fig_top(bc, L, N)
    dum_Tf_dt = get_dum_Tf_dt(Tprop, dtprop)
    return path_fig_top/f'{psi0_str}_{dum_Tf_dt}'


def get_path_fig_top_dyn_multi(bc, L, N, psi0s_str, Tprop, dtprop):
    path_fig_top = get_path_fig_top(bc, L, N)
    dum_Tf_dt = get_dum_Tf_dt(Tprop, dtprop)
    return path_fig_top/f'scan_psi0_{psi0s_str}_{dum_Tf_dt}'


def get_dum_Tf_dt(Tprop, dtprop):
    "Determine string which specifies the propagation limits"
    if Tprop % 1 != 0:
        raise ValueError(f'Tprop={Tprop} has to be integer!')
    dum_Tf_dt = f'Tf_{round(Tprop)}'
    if dtprop != 0.1:
        if (dtprop - round(dtprop, 3)) != 0:
            err_msg = f'self.dtprop={dtprop} only 3 digits allowed!'
            raise ValueError(err_msg)
        dtprop_ = f'{round(dtprop, 3)}'.replace('.', '_')
        dum_Tf_dt += f'_dt_{dtprop_}'
    return dum_Tf_dt


def get_dum_name_U_theta(U, theta, J=None):
    '''Determine part of figure name specifying U, theta, J, if J is None
    assume J=1 and don't mention it. If J=0 and U=1, use U='inf'
    '''
    theta_ = f'{theta/math.pi:.3f}'.replace('.', '_')
    U_ = f'{U:.3f}'.replace('.', '_')
    if J is not None and J != 1:
        if J == 0 or U == 1:
            U_theta_ = f'U_inf_thpi_{theta_}'
        else:
            err_msg = f'U={U}, J={J}; J should be 0 and U=1 for U=inf!'
            raise ValueError(err_msg)
    else:
        U_theta_ = f'U_{U_}_thpi_{theta_}'
    return U_theta_


def get_path_basis_multi_dyn(bc, L, N, U, theta, J, psi0s_str, Tprop, dtprop):
    path_basis = get_path_basis(bc, L, N, U, theta, J)
    dum_Tf_dt = get_dum_Tf_dt(Tprop, dtprop)
    path_npz_dir = path_basis/f'multi_psi0_{psi0s_str}_{dum_Tf_dt}'
    os.makedirs(path_npz_dir, exist_ok=True)
    return path_npz_dir


def get_path_basis_optimize(bc, L, N, U, theta, J, psi0_str, psifinal_str,
                            N_interp=None, Tprop=None, ini_guess=None,
                            version=None):



    path_basis = get_path_basis(bc, L, N, U, theta, J)
    path_basis = path_basis/f'optimize_psi0_{psi0_str}_psifin_{psifinal_str}'

    dir_name = []
    if Tprop is not None:
        Tprop_ = f'Tprop_{Tprop:.1f}'.replace('.', '_')
        dir_name.append(Tprop_)
    if N_interp is not None:
        dir_name.append(f'N_interp_{N_interp}')
    if ini_guess is not None:
        ini_guess_ = f'ini_guess_{ini_guess:.1f}'.replace('.', '_')
        dir_name.append(ini_guess_)
    if version is not None:
        dir_name.append(f'v{version}')
    dir_str = '_'.join(dir_name)

    path_basis = path_basis/dir_str
    os.makedirs(path_basis, exist_ok=True)
    return path_basis


def get_path_fig_optimize(bc, L, N, U_ini, theta_ini, psi0_str, psifinal_str):
    path_fig_top = get_path_fig_top(bc, L, N)
    theta_ = f'{theta_ini/math.pi:.1f}'.replace('.', '_')
    U_ = f'{U_ini:.1f}'.replace('.', '_')
    U_theta_ = f'U_{U_}_thpi_{theta_}'
    path_fig_top = path_fig_top/f'optimze_{psifinal_str}/{psi0_str}_{U_theta_}'
    os.makedirs(path_fig_top, exist_ok=True)
    return path_fig_top


def get_optimize_dum_name(Tprop, version, N_interp, ini_guess):
    Tprop_ = f'Tprop_{Tprop:.1f}'.replace('.', '_')
    ini_guess_ = f'v{version}_ini_guess_{ini_guess:.3f}'.replace('.', '_')
    return f'{Tprop_}_N_interp_{N_interp}_{ini_guess_}'
