import numpy as np
import scipy.special as sspec
import scipy.stats as sstat

def vcol(v):    
    """
    transform v to a col vector and exploit broadcasting
    """    
    return v.reshape((v.shape[0], 1))

def vrow(v):
    """
    transform v to a row vector and exploit broadcasting
    """    
    return v.reshape((1, v.shape[0]))

def load(file_name):
    """load the file "file_name".

    Args:
        file_name (string): path of the file to load.

    Returns:
        dataset, datalabels (np arrays):
            - dataset contains samples and has (#featuresm #samples) shape.
            - datalabels contains labels associated to dataset has a (#samples, ) shape.
    """    
    dataset = []
    datalabels = []
    with open(file_name, 'r') as file:
        for line in file:
            try:
                fields = line.strip().split(',')
                tmp = [[float(x)] for x in fields[:10]]
                dataset.append(np.array(tmp, dtype=np.float32))
                datalabels.append(fields[-1])
            except:
                pass
    dataset = np.hstack(dataset)
    datalabels = np.array(datalabels, dtype=np.int32)
    return dataset, datalabels

def split_db_2to1(D, L, seed=0):
    """split data 2/3 for training and 1/3 for evaluation.

    Args:
        D: dataset
        L: datalabels
        seed (int, optional): used to initialize np.random and perform permutation on data. Defaults to 0.

    Returns:
        (DTR, LTR), (DTE, LTE): two tuples containing samples and labels, one for training and the other evaluation.
    """    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def shuffle(D, L, seed = 0):
    """_summary_
        shuffles dataset D with relative labels L
    Args:
        D (_type_): dataset
        L (_type_): labels
        seed (_type_, optional): seed for random permutation, if None the seed is not set. Defaults to 0.

    Returns:
        _type_: shuffled dataset and labels
    """    
    np.random.seed(seed)
    perm = np.random.permutation(D.shape[1])
    D = D[:, perm]
    L = L[perm]
    return D, L

def compute_correlation(X, Y):
    x_sum = np.sum(X)
    y_sum = np.sum(Y)

    x2_sum = np.sum(X ** 2)
    y2_sum = np.sum(Y ** 2)

    sum_cross_prod = np.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr

def center(D):
    mu = vcol(D.mean(1))
    DC = D - mu
    return DC

def expand(x):
    x = vcol(x)
    tmp = np.dot(x,x.T)
    stacked = np.hstack(tmp.T)
    return vcol(stacked)

def z_norm(D):
    mu = vcol(D.mean(1))
    std = vcol(np.std(D, axis=1))
    return (D-mu)/std

def z_norm_train_test(D1, D2):
    mu = vcol(D1.mean(1))
    std = vcol(np.std(D1, axis=1))
    return (D1-mu)/std, (D2-mu)/std

def cartesian_to_polar(D):
    d1 = D[:5, :]
    d2 = D[5:, :]
    D = d1-d2
    x, y, z, w, v = D[0, :], D[1, :], D[2, :], D[3, :], D[4, :]
    r = np.sqrt(x**2 + y**2 + z**2 + w**2 + v**2)
    theta1 = np.arctan2(y, x)
    theta2 = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
    theta3 = np.arccos(w / np.sqrt(x**2 + y**2 + z**2 + w**2))
    theta4 = np.arccos(v / np.sqrt(x**2 + y**2 + z**2 + w**2 + v**2))
    D = np.array([r, theta1, theta2, theta3, theta4])
    print(D.shape)
    return D

# def atransform_feature(D, feature_idx):
#     def f(x):
#         return np.log(x)-np.log(1-x)
#         # division = np.divide(x, 1-x)
#         # print(division)
#         # np.log(division)
#     eps = 1e-3#np.finfo(float).eps
#     xmin = D[feature_idx].min()
#     xmax = D[feature_idx].max()
    
#     D[feature_idx] = (1-eps) * (D[feature_idx] - xmin) / (xmax-xmin) + eps
#     # D[feature_idx] = np.clip(D[feature_idx], np.finfo(float).eps, 1-np.finfo(float).eps)
#     D[feature_idx] = sspec.logit(D[feature_idx])
#     # print(sorted(D[feature_idx, :]))
#     # print(D[feature_idx, :].shape)
#     return D

# def transform_feature(DTR, TO_GAUSS):
#     P = []
#     for dIdx in range(DTR.shape[0]):
#         DT = vcol(TO_GAUSS[dIdx, :])
#         X = DTR[dIdx, :] < DT
#         R = (X.sum(1) + 1) / (DTR.shape[1] + 2)
#         P.append(sstat.norm.ppf(R))
#     return np.vstack(P)
