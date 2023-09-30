import numpy
import scipy
from .MVG import MultivariateGaussianClassifier

labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
def load(file_name):
    dataset = []
    datalabels = []
    with open(file_name, 'r') as file:
        for line in file:
            try:
                fields = line.strip().split(',')
                tmp = [[float(x)] for x in fields[:4]]
                dataset.append(numpy.array(tmp, dtype=numpy.float32))
                datalabels.append(labels[fields[4]])    
            except:
                pass
    dataset = numpy.hstack(dataset)
    datalabels = numpy.array(datalabels, dtype=numpy.int32)
    return [dataset, datalabels]

def shuffle(D, L, seed = 0):
    numpy.random.seed(seed)
    tmp = [(D[:,i], L[i]) for i in range(D.shape[1])]
    numpy.random.shuffle(tmp)
    D = numpy.array([x[0] for x in tmp]).T
    L = numpy.array([x[1] for x in tmp])
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def vcol(v):    # transform v to a col vector and exploit broadcasting
    return v.reshape((v.shape[0], 1))

def vrow(v):    # transform v to a row vector and exploit broadcasting
    return v.reshape((1, v.shape[0]))
    
def ML_estimate(X):
    N = X.shape[1]
    mu = vcol(X.mean(1))
    XC = X - mu
    C = numpy.dot(XC,XC.T) / N
    return mu, C

def logpdf_GAU_ND(X, mu, C):
    Y = []
    M = X.shape[0]
    #print(M)
    for x in X.T:
        x = vcol(x)
        a = M*numpy.log(2*numpy.pi)
        b = numpy.linalg.slogdet(C)[1]
        c = numpy.dot(numpy.dot((x-mu).T, numpy.linalg.inv(C)), (x-mu))[0,0]
        Y.append(-0.5*(a+b+c))
    return numpy.array(Y)

def profsol():
    D, L = load('iris.csv')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    hCls = {}
    for lab in [0,1,2]:
        DCLS = DTR[:, LTR==lab]
        hCls[lab] = ML_estimate(DCLS)
    prior = vcol(numpy.ones(3)/3.0)
    S = []
    for hyp in [0,1,2]:
        mu, C = hCls[hyp]
        fcond = numpy.exp(logpdf_GAU_ND(DTE, mu, C))
        S.append(vrow(fcond))
    S = numpy.vstack(S)
    S = S*prior
    SJoint_sol = numpy.load('Solution/SJoint_MVG.npy')
    print((S-SJoint_sol).max())
    P = S/vrow(S.sum(0))

def iris_classification(DTR, LTR, DTE, LTE, naive_bayes = False, tied = False):
    mCs = {}
    labels_unique = set(LTR)
    wcC = []
    for lab in labels_unique:
        m, C = ML_estimate(DTR[:, LTR==lab])
        if naive_bayes:
            identity = numpy.identity(C.shape[0])
            C *= identity
        mCs[lab] = (m, C)
        if tied:
            if len(wcC) == 0:
                wcC = DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
            else:
                wcC += DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
    
    if tied:
        for k in mCs.keys():
            mCs[k] = (mCs[k][0], wcC)
    
    #print(mCs)
    S = []
    for lab in labels_unique:
        m, C = mCs[lab]
        fcond = numpy.exp(logpdf_GAU_ND(DTE, m, C))
        S.append(vrow(fcond))
    S = numpy.vstack(S)

    Pc = vcol(numpy.ones(len(labels_unique))/len(labels_unique))
    SJoint = S*Pc
    marginals = vrow(SJoint.sum(0))
    SPost = SJoint / marginals

    # SJoint_sol = numpy.load('Solution/SJoint_TiedNaiveBayes.npy')
    # print("Error wrt sol: " + str((SJoint-SJoint_sol).max()))

    Lpred = numpy.argmax(SPost, 0)
    # correct = (Lpred == LTE).sum()
    # acc = correct / LTE.shape[0]
    # err = 1-acc
    # print(f'Acc = {acc*100}%')
    # print(f'Error rate = {(err*100.0):.1f}%')
    return Lpred

def iris_log_classification(DTR, LTR, DTE, LTE, naive_bayes = False, tied = False):
    mCs = {}
    labels_unique = set(LTR)
    wcC = []
    for lab in labels_unique:
        m, C = ML_estimate(DTR[:, LTR==lab])
        if naive_bayes:
            identity = numpy.identity(C.shape[0])
            C *= identity
        mCs[lab] = (m, C)
        if tied:
            if len(wcC) == 0:
                wcC = DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
            else:
                wcC += DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
    
    if tied:
        for k in mCs.keys():
            mCs[k] = (mCs[k][0], wcC)

    logS = []
    for lab in labels_unique:
        m, C = mCs[lab]
        logS.append(vrow(logpdf_GAU_ND(DTE, m, C)))
    logS = numpy.vstack(logS)
    
    logPc = vcol(numpy.log(numpy.ones(len(labels_unique))/len(labels_unique)))
    logSJoint = logPc + logS
    logSmarginals = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    # print(logSmarginals)
    logSPost = logSJoint - logSmarginals
    SPost = numpy.exp(logSPost)
    
    # logSJoint_sol = numpy.load('Solution/logSJoint_TiedNaiveBayes.npy')
    # print("Error wrt sol: " + str((logSJoint-logSJoint_sol).max()))

    Lpred = numpy.argmax(SPost, 0)
    # correct = (Lpred == LTE).sum()
    # acc = correct / LTE.shape[0]
    # err = 1-acc
    # print(f'Acc = {acc*100}%')
    # print(f'Error rate = {(err*100.0):.1f}%')
    return Lpred

def Kfold(D, L, K, model, naiveBayes = False, tied = False):
    D, L = shuffle(D, L)
    N = D.shape[1]
    Lpred = []
    for i in range(0, D.shape[1], N//K):
        idxTrain = numpy.concatenate((numpy.arange(i), numpy.arange(i+N//K, D.shape[1])))
        idxTest = numpy.arange(i, min(i+N//K, N))
        #print(idxTrain, idxTest)
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        # print(DTR.shape, DTE.shape, LTR.shape, LTE.shape)
        tmp = model(DTR, LTR, DTE, LTE, naiveBayes, tied)
        for p in tmp:
            Lpred.append(p)
    #print('Lpred = ', Lpred)
    correct = (Lpred == L).sum()
    acc = correct / L.shape[0]
    err = 1-acc
    print(f'Accuracy    =  {acc*100:.1f}%')
    print(f'Error rate  =  {err*100:.1f}%')


def LOO(D, L, model, naiveBayes = False, tied = False):
    Lpred = []
    for i in range(D.shape[1]):
        idxTrain = numpy.concatenate((numpy.arange(i), numpy.arange(i+1, D.shape[1])))
        idxTest = numpy.array([i])
        #print(idxTrain, idxTest)
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        tmp = model(DTR, LTR, DTE, LTE, naiveBayes, tied)
        #print(tmp)
        Lpred.append(tmp[0])
    correct = (Lpred == L).sum()
    acc = correct / L.shape[0]
    err = 1-acc
    print(f'Accuracy    =  {acc*100:.1f}%')
    print(f'Error rate  =  {err*100:.1f}%')

def LOO2(D, L, naiveBayes = False, tied = False):
    Lpred = []
    for i in range(D.shape[1]):
        idxTrain = numpy.concatenate((numpy.arange(i), numpy.arange(i+1, D.shape[1])))
        idxTest = numpy.array([i])
        #print(idxTrain, idxTest)
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        model = MultivariateGaussianClassifier(naiveBayes, tied)
        model.train(DTR, LTR)
        tmp = model.test(DTE)
        #print(tmp)
        Lpred.append(tmp[0])
    correct = (Lpred == L).sum()
    acc = correct / L.shape[0]
    err = 1-acc
    print(f'Accuracy    =  {acc*100:.1f}%')
    print(f'Error rate  =  {err*100:.1f}%')


# iris_classification(True, True)
# iris_log_classification(True, True)
def run():
    D, L = load('iris.csv')
    LOO(D, L, iris_log_classification, False, False)
    LOO2(D, L, False, False)
    LOO(D, L, iris_log_classification, True, False)
    LOO2(D, L, True, False)
    LOO(D, L, iris_log_classification, False, True)
    LOO2(D, L, False, True)
    LOO(D, L, iris_log_classification, True, True)
    LOO2(D, L, True, True)