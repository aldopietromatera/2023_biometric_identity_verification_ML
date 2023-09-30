import numpy as np

def Kfold(D, L, K, func):
    """_summary_
        Runs K-fold on dataset

    Args:
        D (_type_): dataset
        L (_type_): labels
        K (_type_): number of folds
        func (_type_): should accept DTR, LTR, DTE, LTE as parameter and specify what to do with them
    """
    N = D.shape[1]
    cnt = 0
    for i in range(0, D.shape[1], N//K):
        idxTrain = np.concatenate((np.arange(i), np.arange(i+N//K, D.shape[1])))
        idxTest = np.arange(i, min(i+N//K, N))
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        func(cnt, DTR, LTR, DTE, LTE)
        cnt += 1
    
