import math
import numpy as np

def time_str(sec):
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return f'{int(hour)}h:{int(min)}m:{sec:.2f}s'


def add_to_set(input_set, input_val, threshold):
    """Add input_val to input_set if no item in input_set is close to input_val

    Args:
        input_set (set): set
        input_val (float): val
        threshold (float): limit

    Returns:
        bool : whether input_val has been added or not
    """

    nearest_list = [el for el in input_set if abs(input_val - el) < threshold]
    if len(nearest_list) == 0:
        input_set.add(input_val)
        return True
    else:
        assert len(nearest_list) == 1
        return False


def find_degeneracies(eval_in):
    """Determine the degeneracies of a given input list.

    Returns
    dict : degeneracies
    """
    dict_energies = {}

    # energy_set = set(self.evals[0])
    # state_ind_list = [0]
    # energy_prev = str(self.evals[0])
    energy_set = set([eval_in[0]])
    state_ind_list = []
    energy_prev = ''

    for i, energy in enumerate(eval_in):

        bool_add = add_to_set(energy_set, energy, 1e-10)

        if bool_add:
            dict_energies[energy_prev] = state_ind_list

            # initialize new
            state_ind_list = [i]

        else:
            state_ind_list.append(i)

        energy_prev = str(energy)

    dict_energies[energy_prev] = state_ind_list

    return dict_energies

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


def make_schmidt_decomp(bipartitepurestate_tensor):
    """https://github.com/stephenhky/pyqentangle/blob/master/pyqentangle/
    schmidt.py"""
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)
    vecs1, diags, vecs2_h = np.linalg.svd(bipartitepurestate_tensor)
    vecs2 = vecs2_h.transpose()
    decomposition = [(diags[k], vecs1[:, k], vecs2[:, k])
        for k in range(mindim)]
    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)
    return decomposition
