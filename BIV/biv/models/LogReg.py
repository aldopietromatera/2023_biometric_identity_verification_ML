import numpy as np
import scipy.optimize as sciopt
from .IModel import IModel

class BinaryLogReg(IModel):
    """Binary Logistic Regression model implementation"""    
    def __init__(self, l=0.0, prior=None):
        """Model constructor

        Args:
            l (float, optional): lambda parameter for regulation term. Defaults to 0
        """
        self.l = l
        self.prior = prior
    
    def _logreg_obj(self, v):
        """
            Objective function to minimize for Logistic Regression
        """
        w, b = v[0:-1], v[-1]
        z = 2*self.L-1
        s = np.dot(w.T, self.D)+b
        mean = np.average(np.logaddexp(0, -z*s))
        reg_term = 0.5*self.l*np.linalg.norm(w)**2
        return reg_term + mean
    
    def _logreg_obj_prior_weighted(self, v):
        """
            Objective function to minimize for Logistic Regression
        """
        w, b = v[0:-1], v[-1]
        z = 2*self.L-1
        s = np.dot(w.T, self.D)+b
        loss = np.logaddexp(0, -z*s)
        avg_1 = np.sum(loss[z==1])*self.prior/(z==1).sum()
        avg_2 = np.sum(loss[z==-1])*(1-self.prior)/(z==-1).sum()
        reg_term = 0.5*self.l*np.linalg.norm(w)**2
        return reg_term + avg_1 + avg_2
    
    def _logreg_obj_with_gradient(self, v):
        """
            Objective function to minimize for Logistic Regression, returning also the gradient
        """
        w, b = v[0:-1], v[-1]
        z = 2*self.L-1
        s = np.dot(w.T, self.D)+b
        mean = np.average(np.logaddexp(0, -z*s))
        reg_term = 0.5*self.l*np.linalg.norm(w)**2
        f = reg_term + mean

        gfrac = np.exp(-z*s)/(1+np.exp(-z*s))
        djdw = np.average(gfrac*(-z*self.D), axis = 1) + self.l*w
        djdb = np.average(gfrac*(-z))
        grad = np.hstack([djdw, djdb])
        print('f',f,'grad',grad)
        return f, grad


    def train(self, DTR, LTR):
        """Method to train the model

        Args:
            DTR: training dataset with samples stacked as column vectors
            LTR: training labels
        """        
        self.D = DTR
        self.L = LTR

        x0 = np.zeros(self.D.shape[0]+1)
        v_opt, _, _ = sciopt.fmin_l_bfgs_b(self._logreg_obj_prior_weighted, x0, approx_grad=True)
        w, b = v_opt[0:-1], v_opt[-1]
        self.w = w
        self.b = b
    
    def test(self, DTE):
        """Method to test the model (requires training before)

        Args:
            DTE: test dataset

        Raises:
            AttributeError: missing training data. Call train before test

        Returns:
            A numpy array with predicted labels
        """        
        if self.w is None or self.b is None:
            raise AttributeError()
        scores = np.dot(self.w.T, DTE) + self.b
        return scores > 0, scores
    
    @staticmethod
    def load(file_name):
        raise NotImplementedError()