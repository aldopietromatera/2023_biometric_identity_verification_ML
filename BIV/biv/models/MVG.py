import numpy as np
import scipy.special as ss
from utils import utils
from .IModel import IModel

def _logpdf_GAU_ND(x, mu, C):
    """Function to compute the probability of x given a gaussian distribution with mean mu and covariance matrix C.

    Args:
        x: dataset
        mu: density mean
        C: density covariance matrix

    Returns:
        for each sample, the probability for it to occur
    """
    XC = x - mu
    M = x.shape[0]
    const = - 0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC*np.dot(L, XC)).sum(0)

    return const - 0.5 * logdet - 0.5*v

def _ML_estimate(X):
    """Function to compute optimal values for a gaussian distribution modeling dataset X

    Args:
        X: dataset

    Returns:
        mu, C (mean and covariance matrix of the dataset)
    """    
    N = X.shape[1]
    mu = utils.vcol(X.mean(1))
    XC = X - mu
    C = np.dot(XC,XC.T) / N
    return mu, C

# @DeprecationWarning()
class MultivariateGaussianClassifier(IModel):
    """Multivariate Gaussian Classifier model implementation"""    
    def __init__(self, naive_bayes = False, tied = False):
        """Initialize the MVG classifier

        Args:
            naive_bayes (bool, optional): specify if naive assumption should be used. Defaults to False.
            tied (bool, optional): specify if the data has a tied covariance matrix. Defaults to False.
        """
        self.__naive_bayes = naive_bayes
        self.__tied = tied        
        
    def train(self, DTR, LTR, save_file_name = None):
        """Method to train the model

        Args:
            DTR: training dataset
            LTR: training dataset labels
        """        
        self.__mCs = {}
        self.__labels_unique = set(LTR)
        wcC = []
        for lab in self.__labels_unique:
            m, C = _ML_estimate(DTR[:, LTR==lab])
            if self.__naive_bayes:
                identity = np.identity(C.shape[0])
                C *= identity
            self.__mCs[lab] = (m, C)
            if self.__tied:
                if len(wcC) == 0:
                    wcC = DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
                else:
                    wcC += DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
        if self.__tied:
            for k in self.__mCs.keys():
                self.__mCs[k] = (self.__mCs[k][0], wcC)
        
        if save_file_name:
            raise NotImplementedError("Not implemented for this model")

    @staticmethod
    def load(file_name):
        raise NotImplementedError("Not implemented for this model")
        model = MultivariateGaussianClassifier()
        data = np.load(file_name, allow_pickle=True)
        
        model.naive_bayes = data[0]
        model.tied = data[1]
        model.__labels_unique = data[2]
        model.__mCs = data[3]
        return model

    def test(self, DTE):
        """Method to test the model (requires training before)

        Args:
            DTE: test dataset

        Raises:
            AttributeError: missing training data. Call train before test

        Returns:
            numpy array with predicted labels, numpy array with LLRs
        """        
        if self.__mCs is None:
            raise AttributeError()
        logS = []
        for lab in self.__labels_unique:
            m, C = self.__mCs[lab]
            logS.append(utils.vrow(_logpdf_GAU_ND(DTE, m, C)))
        logS = np.vstack(logS)
        
        logPc = utils.vcol(np.log(np.ones(len(self.__labels_unique))/len(self.__labels_unique)))
        logSJoint = logPc + logS
        logSmarginals = utils.vrow(ss.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSmarginals
        Lpred = np.argmax(logSPost, 0)

        LLRs = logS[1, :] - logS[0, :]
        
        return Lpred, LLRs

class BinaryMultivariateGaussianClassifier(IModel):
    """Multivariate Gaussian Classifier model implementation"""    
    def __init__(self, naive_bayes = False, tied = False):
        """Initialize the MVG classifier

        Args:
            naive_bayes (bool, optional): specify if naive assumption should be used. Defaults to False.
            tied (bool, optional): specify if the data has a tied covariance matrix. Defaults to False.
        """        
        self.__naive_bayes = naive_bayes
        self.__tied = tied
        
    def train(self, DTR, LTR, save_file_name = None):
        """Method to train the model

        Args:
            DTR: training dataset
            LTR: training dataset labels
        """        
        empirical_priors = ((LTR.shape[0]-LTR.sum())/LTR.shape[0], LTR.sum()/LTR.shape[0])
        self.__treshold = -np.log(empirical_priors[0]/empirical_priors[1])
        
        self.__mCs = {}
        wcC = None
        for lab in {0, 1}:
            m, C = _ML_estimate(DTR[:, LTR==lab])
            if self.__naive_bayes:
                C *= np.identity(C.shape[0])
            self.__mCs[lab] = (m, C)
            if self.__tied:
                if wcC is None:
                    wcC = DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
                else:
                    wcC += DTR[:, LTR==lab].shape[1] / DTR.shape[1] * C
        if self.__tied:
            for lab in {0, 1}:
                self.__mCs[lab] = (self.__mCs[lab][0], wcC)
    
        if save_file_name:
            np.save(save_file_name, np.array([self.__naive_bayes, self.__tied, self.__mCs, self.__treshold]))

    def test(self, DTE):
        """Method to test the model (requires training before)

        Args:
            DTE: test dataset

        Raises:
            AttributeError: missing training data. Call train before test

        Returns:
            numpy array with predicted labels, numpy array with LLRs
        """
        if self.__mCs is None:
            raise AttributeError()
        den = _logpdf_GAU_ND(DTE, self.__mCs[0][0], self.__mCs[0][1])
        num = _logpdf_GAU_ND(DTE, self.__mCs[1][0], self.__mCs[1][1])
        llrs = num-den
        return llrs > self.__treshold, llrs
    
    @staticmethod
    def load(file_name):
        tmp = np.load(file_name, allow_pickle=True)
        model = BinaryMultivariateGaussianClassifier()
        model.__naive_bayes = tmp[0]
        model.__tied = tmp[1]
        model.__mCs = tmp[2]
        model.__treshold = tmp[3]
        return model
        #raise NotImplementedError("Not implemented for this model")
