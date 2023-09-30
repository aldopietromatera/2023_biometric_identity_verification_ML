import numpy

from utils import utils

def PCA(D, m, ret_eigs=False):
    """perform dimensionality reduction with PCA (unsupervised)

    Args:
        D : dataset
        m : number of the dimension of the subspace of the original space

    Returns:
        original data projected on the subspace
    """    
    mu = utils.vcol(D.mean(1))
    DC = D - mu
    C = numpy.dot(DC, DC.T) / D.shape[1]

    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, :m]
    s = s[::-1] # sort eigvals in descending order
    # print(P)

    DP = numpy.dot(P.T, D)
    return DP, s if ret_eigs else DP

def PCA_train_test(D1, D2, m):
    mu = utils.vcol(D1.mean(1))
    DC = D1 - mu
    C = numpy.dot(DC, DC.T) / D1.shape[1]

    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, :m]
    s = s[::-1] # sort eigvals in descending order
    # print(P)

    DP1 = numpy.dot(P.T, D1)
    DP2 = numpy.dot(P.T, D2)
    return DP1, DP2