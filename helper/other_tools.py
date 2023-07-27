import math


def get_path_basis(path_data_top, L, N, U, theta):
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

    theta_ = f'{theta/math.pi:.3f}'.replace('.', '_')
    U_ = f'{U:.3f}'.replace('.', '_')
    path_basis = path_data_top/f'L{L}_N{N}/U_{U_}_thpi_{theta_}'
    path_basis.mkdir(parents=True, exist_ok=True)

    return path_basis


# def get_path_prop(path_data_top, L, N, U, theta, psi_ini):
#     """Create path of data from input pars. J is assumed to be always 1
#
#     Args:
#         path_data_top (Path): top path of data folder
#         L (int): number of lattice sites
#         N (int): number of atoms
#         U (float): on-site interaction
#         theta (float): statistical angle
#         psi_ini (str): initial state
#
#     Returns:
#         path_prop (Path): path where propagation is stored
#     """
#
#     path_basis = get_path_basis(path_data_top, L, N, U, theta)
#     path_prop = path_basis/f'{psi_ini}'
#     path_prop.mkdir(parents=True, exist_ok=True)
#
#     return path_prop
