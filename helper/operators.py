import math
import numpy as np


def sum_hop(L, N, theta, basis1, basis2):
    "sum_j^L-1 b_j^t exp(i*theta*n_j) b_j+1"

    sum_hop = 0
    for j in range(L-1):
        b1 = basis1[::]
        b2 = basis2[::]

        # b_j+1
        if b2[j+1] == 0:
            continue
        else:
            hop_jp1 = math.sqrt(b2[j+1])
            b2[j+1] -= 1

        # exp(i * theta * n_j)
        exp_j = np.exp(1j*theta*b2[j])

        # b_j^t
        if b2[j] == N:
            continue
        else:
            hop_j = math.sqrt(b2[j]+1)
            b2[j] += 1
        if b1 == b2:
            sum_hop += hop_j*exp_j*hop_jp1

    return sum_hop


#-----------------------------------------------------------------------
def sum_hop_t(L, N, theta, basis1, basis2):
    "sum_j^L-1 b_j+1^t exp(-i*theta*n_j) b_j"

    sum_hop = 0
    for j in range(L-1):
        b1 = basis1[::]
        b2 = basis2[::]

        # b_j
        if b2[j] == 0:
            continue
        else:
            hop_j = math.sqrt(b2[j])
            b2[j] -= 1

        # exp(-i * theta * n_j)
        exp_j = np.exp(-1j*theta*b2[j])

        # b_j+1^t
        if b2[j+1] == N:
            continue
        else:
            hop_jp1 = math.sqrt(b2[j+1]+1)
            b2[j+1] += 1

        if b1 == b2:
            sum_hop += hop_j*exp_j*hop_jp1

    return sum_hop


#-----------------------------------------------------------------------
def sum_onsite_int(L, basis1, basis2):
    "U/2 sum_j^L  n_j(n_j-1)"

    if basis1 != basis2:
        return 0

    sum_int = 0
    for j in range(L):

        if basis1[j] == 0:
            continue
        else:
            sum_int += basis1[j]*(basis1[j] - 1)

    return sum_int


#-----------------------------------------------------------------------
def bi_dagger_bj(basis1, basis2, i, j, N):
    "<1,2,0,1| b_i^t b_j | 2,2,0,0>"


    b1 = basis1[::]
    b2 = basis2[::]

    # b_j |2>
    if b2[j] == 0:
        return 0
    else:
        bj = math.sqrt(b2[j])
        b2[j] -= 1

    # b_i^t b_j |2>
    if b2[i] == N:
        return 0
    else:
        bi_dagger = math.sqrt(b2[i] + 1)
        b2[i] += 1

    if b1 == b2:
        return bi_dagger*bj
    else:
        return 0


#-----------------------------------------------------------------------
def K_operator(L, theta, basis1, basis2):
    """ symmetry operator
        PRL 118, 120401 (2017)
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.118.120401
        Eq. (12)

        K = exp(i\theta sum_{j=1}^L n_j(n_j-1)/2) * I * T

        I : inversion operator: |1, 2, 3, 4>  -->  |4, 3, 2, 1>
        T : time-reversal op: np.conjugate(|1, 2, 3, 4>)

        Apply time-reversal operator separately!
    """

    b1 = basis1[::]
    b2 = basis2[::]


    # inversion
    b2 = b2[::-1]

    if np.allclose(b1, b2):
        sum_njnj = sum([b2[j]*(b2[j] - 1) for j in range(L)])
        exp_term = np.exp(1j*theta*sum_njnj/2)
        return exp_term

    else:
        return 0


def K_dagger_operator(L, theta, basis1, basis2):
    """ symmetry operator
        PRL 118, 120401 (2017)
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.118.120401
        Eq. (12)

        K = exp(i\theta sum_{j=1}^L n_j(n_j-1)/2) * I * T

        I : inversion operator: |1, 2, 3, 4>  -->  |4, 3, 2, 1>
        T : time-reversal op: np.conjugate(|1, 2, 3, 4>)

        Apply time-reversal operator separately!
    """

    b1 = basis1[::]
    b2 = basis2[::]


    # inversion
    b1 = b1[::-1]

    if np.allclose(b1, b2):
        sum_njnj = sum([b2[j]*(b2[j] - 1) for j in range(L)])
        exp_term = np.exp(-1j*theta*sum_njnj/2)
        return exp_term

    else:
        return 0




#-----------------------------------------------------------------------
def pair_operator(L, basis1, basis2):
    """ \nu_p = \sum_j \hat{n}_j (\hat{n}_j - 1)/2  """

    if np.allclose(basis1, basis2):
        sum_njnj = sum([basis1[j]*(basis1[j] - 1)/2 for j in range(L)])
        return sum_njnj

    else:
        return 0

#===============================================================================
# momentum operator
#===============================================================================
def get_bibj_correlator(psi, basis_list, i, j):
    r"""\langle b_i^\dagger b_j \rangle
        = <psi | b_i^\dagger b_j | psi >
        = (<4,0,0,0|c_1 + <3,1,0,0|c_2 + ... ) | b_i^\dagger b_j | psi >
        """
    N = sum(basis_list[0])
    bibj_op = 0
    for ind_1, basis1 in enumerate(basis_list):
        for ind_2, basis2 in enumerate(basis_list):
            # apply bi^dagger b_j
            coeff_bibj = bi_dagger_bj(basis1, basis2, i, j, N)

            if coeff_bibj == 0:
                continue
            else:
                bibj_op += coeff_bibj*np.conjugate(psi[ind_1])*psi[ind_2]

    return bibj_op
