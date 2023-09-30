import numpy
import scipy.linalg

from utils import utils

def compute_Sw_Sb(D, L):
    num_classes = L.max()+1
    D_c = [D[:, L==i] for i in range(num_classes)]
    n_c = [D_c[i].shape[1] for i in range(num_classes)]
    mu = utils.vcol(D.mean(1))
    mu_c = [utils.vcol(D_c[i].mean(1)) for i in range(len(D_c))]
    S_w, S_b = 0, 0
    for i in range(num_classes):
        DC = D_c[i] - mu_c[i]
        C_i = numpy.dot(DC, DC.T) / DC.shape[1]
        S_w += n_c[i] * C_i
        diff = mu_c[i] - mu
        S_b += n_c[i] * numpy.dot(diff, diff.T)
    S_w /= D.shape[1]
    S_b /= D.shape[1]
    # print(S_w)
    # print(S_b)
    return S_w, S_b

def project(base, D):
    return numpy.dot(base.T, D)

def LDA(D, L, m):
    """perform dimensionality reduction with LDA (supervised)

    Args:
        D : dataset
        L:  labels
        m : number of the dimension of the subspace of the original space (m must be smaller than #classes-1)

    Returns:
        original data projected on the subspace
    """    
    S_w, S_b = compute_Sw_Sb(D, L)
    _, U = scipy.linalg.eigh(S_b, S_w)
    W = U[:, ::-1][:, :m]
    DP = project(W, D)
    return DP
