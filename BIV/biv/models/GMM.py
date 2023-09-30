import numpy as np
import scipy.special as ss
from utils import utils
from .IModel import IModel
from .MVG import _logpdf_GAU_ND, _ML_estimate

def logpdf_GMM(X, gmm):
    joints = []
    for w, mu, C in gmm:
        joints.append(_logpdf_GAU_ND(X, mu, C) + np.log(w))
    S = np.vstack(joints)
    logdens = ss.logsumexp(S, axis=0)
    return logdens

def eigs_constraint(C, psi):
    U, s, _ = np.linalg.svd(C)
    s[s<psi] = psi
    return np.dot(U, utils.vcol(s)*U.T)

def EM(D, GMM, psi=0.1, diagonal=False, tied=False):
    DELTA = 1e-6
    maxiter = 1000
    pre_l = -1e100
    G = len(GMM) # the number of clusters
    N = D.shape[1] # the number of samples
    while maxiter > 0:
        maxiter -= 1
        joints = []
        for w, mu, C in GMM:
            joints.append(_logpdf_GAU_ND(D, mu, C) + np.log(w))
        joints = np.vstack(joints)
        marginals = ss.logsumexp(joints, axis=0)
        resps = np.exp(joints - marginals)
        l = marginals.sum()/N
        if l - pre_l < DELTA:
            break
        pre_l = l

        # compute zero, first & second order statistics
        Z = resps.sum(1)
        F = np.dot(resps, D.T)
        S = []
        for i in range(G):
            S.append(np.dot(resps[i]*D, D.T))
        S = np.asarray(S)

        # use statistic to compute new model parameters
        w = Z / D.shape[1] # Z.sum() is the number of samples (D.shape[1])
        mu = F / utils.vcol(Z)
        C = S / Z.reshape((G, 1, 1)) - mu[:,np.newaxis,:]*mu[:,:,np.newaxis]
        if diagonal:
            C = C * np.eye(C.shape[1])
        if tied:
            Ctied = np.sum(C*Z.reshape((C.shape[0], 1, 1)), 0)/N
            C = np.ones((C.shape[0], 1, 1)) * Ctied
        for i in range(C.shape[0]):
            C[i] = eigs_constraint(C[i], psi)
        GMM = [(w[i], utils.vcol(mu[i]), C[i]) for i in range(G)]
    
    if maxiter == 0:
        print('GMM-EM: Reached max number of iterations')
    return GMM

def LBG(D, K, diagonal=False, tied=False, alpha = 0.1, psi = 0.01):
    mu, C = _ML_estimate(D)
    if diagonal:
        C = C * np.eye(C.shape[0])
    C = eigs_constraint(C, psi)
    GMM = [(1, mu, C)]
    # print('GMM = \n', GMM)

    k = 1
    while k < K:
        newGMM = []
        for w, mu, C in GMM:
            U, s, _ = np.linalg.svd(C)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newGMM.append((w/2, mu-d, C))
            newGMM.append((w/2, mu+d, C))
        GMM = EM(D, newGMM, psi=psi, diagonal=diagonal, tied=tied)
        k *= 2
    return GMM

class GMMClassifier(IModel):
    def __init__(self, tied = False, diagonal = False):
        self.__tied = tied
        self.__diagonal = diagonal
        
    def train(self, DTR, LTR, K):
        self.__mCs = {}
        self.__labels_unique = set(LTR)
        for lab in self.__labels_unique:
            GMM = LBG(DTR[:, LTR==lab], K[lab], diagonal=self.__diagonal, tied=self.__tied)
            self.__mCs[lab] = GMM
        # print('mCs = ', self.__mCs)
            
    def test(self, DTE):
        if self.__mCs is None:
            raise AttributeError()
        logS = []
        for lab in self.__labels_unique:
            GMM = self.__mCs[lab]
            logS.append(utils.vrow(logpdf_GMM(DTE, GMM)))
        logS = np.vstack(logS)

        llrs = logS[1] - logS[0]
        
        logPc = utils.vcol(np.log(np.ones(len(self.__labels_unique))/len(self.__labels_unique)))
        logSJoint = logPc + logS
        logSmarginals = utils.vrow(ss.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSmarginals
        Lpred = np.argmax(logSPost, 0)
        return Lpred, llrs
    
    @staticmethod
    def load(file_name):
        raise NotImplementedError()
