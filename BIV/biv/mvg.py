from utils import load, attribute_hists, shuffle, plot_correlations, split_db_2to1, center, plot_hist_scatter, cartesian_to_polar
import models
from validation import Kfold
from measuring_predictions import compute_actDCF, compute_minDCF_optimized
import numpy as np
from dimred import PCA
import sys

def runKfold(D, L, K):
    Lpred = []
    LLRs = []

    def kfold_operation(cnt, DTR, LTR, DTE, LTE):
        print(f'running k-fold on fold #{cnt}')
        # model = models.MVGClassifier()
        model = models.MVGBinClassifier(tied=False, naive_bayes=True)
        model.train(DTR, LTR)
        _, llrs = model.test(DTE)
        LLRs.append(llrs)
        # for p in tmp:
        #     Lpred.append(p)
        # for l in llrs:
        #     print(l)
        #     LLRs.append(l)

    Kfold(D, L, K, kfold_operation)
    # Lpred = np.array(Lpred).reshape(L.shape)
    return np.array(LLRs).reshape(L.shape)

def save_scores(scores, L, file_name):
    path = 'scores/mvg/uncal/'+file_name
    print(f'saving scores at {path}', file=sys.stderr)
    np.save(path, np.vstack([scores, L]))

def run():
    mPCAs = [None, 6, 7, 8, 9, 10]
    mPCAs = [None, 5, 4, 3, 2, 1]
    # mPCAs = [8]
    for mPCA in mPCAs:
        print('mPCA = ' + str(mPCA))

        print('loading data', file=sys.stderr)
        D, L = load('data/Train.txt')
        # print('center data\n')
        # D = center(D)
        D = cartesian_to_polar(D)
        if mPCA is not None:
            D, _ = PCA(D, mPCA)
        # D = LDA(D, L, 1)

        print('shuffling data', file=sys.stderr)
        D, L = shuffle(D, L)

        print('run K-fold', file=sys.stderr)
        LLRs = runKfold(D, L, K=5)
        save_scores(LLRs, L, f'mvg_naive_scores_{mPCA}.npy')
        # print(LLRs.shape, L.shape)

        # print('computing minDCF - actDCF')
        # list_of_eff_priors = [0.1, 0.5]
        # minDCFs, DCFs = compute_minDCF_optimized(list_of_eff_priors, LLRs, L), compute_actDCF(list_of_eff_priors, LLRs, L)
        # print('list of eff priors = ' + str(list_of_eff_priors))
        # print('minDCF = ' + str(minDCFs))
        # print('actDCF = ' + str(DCFs))
        # print()
