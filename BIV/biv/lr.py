from utils import load, attribute_hists, shuffle, plot_correlations, split_db_2to1, center, plot_hist_scatter, expand, z_norm, vrow, cartesian_to_polar
import models
from validation import Kfold
from measuring_predictions import compute_min_DCF, compute_minDCF_optimized, compute_actDCF
import numpy as np
from dimred import PCA
import sys


def runKfold(D, L, K, model):
    LLRs = []

    def kfold_operation(cnt, DTR, LTR, DTE, LTE):
        print(f'running k-fold on fold #{cnt}', file=sys.stderr)
        model.train(DTR, LTR)
        _, llrs = model.test(DTE)
        LLRs.append(llrs)

    Kfold(D, L, K, kfold_operation)
    # Lpred = np.array(Lpred).reshape(L.shape)
    return np.array(LLRs).reshape(L.shape)

def runKfold_cal(D, L, K):
    print('running calibration K-fold', file=sys.stderr)
    LLRs = []

    def kfold_operation(cnt, DTR, LTR, DTE, LTE):
        print(f'running k-fold on fold #{cnt}', file=sys.stderr)
        model = models.LRBinClassifier(l=0, prior=0.1)
        model.train(DTR, LTR)
        _, llrs = model.test(DTE)
        LLRs.append(llrs)

    Kfold(D, L, K, kfold_operation)
    LLRs = np.array(LLRs).reshape(L.shape)
    LLRs -= np.log(0.1/0.9)
    return LLRs

def run_mindcf(scores, L):
    print('computing minDCF - actDCF', file=sys.stderr)
    list_of_eff_priors = [0.1, 0.5]
    minDCFs, DCFs = compute_minDCF_optimized(list_of_eff_priors, scores, L), compute_actDCF(list_of_eff_priors, scores, L)
    print('list_of_eff_priors = ' + str(list_of_eff_priors))
    print('minDCF = ' + str(minDCFs), file=sys.stderr)
    print('minDCF = ' + str(minDCFs))
    print('actDCF = ' + str(DCFs))


def expand_dataset(D):
    D2 = []
    # print(D.shape)
    for i in range(D.shape[1]):
        D2.append(expand(D[:, i]))
    print(np.hstack(D2).shape)
    return np.hstack(D2)

def transform(D):
    d1 = D[:5, :]
    d2 = D[5:, :]
    D = d1-d2
    # compute the square radius
    D = np.power(D, 2)
    D = vrow(D.sum(axis = 0))
    return D

def run_linear():
    ms = [None, 5, 4, 3, 2, 1]
    for m in ms:
        print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
        print('loading data', file=sys.stderr)
        D, L = load('/home/andrea/polito/MLPR_BIV/BIV/data/Train.txt')
        print('center data', file=sys.stderr)
        D = cartesian_to_polar(D)
        if not m is None:
            D, _ = PCA(D, m)
        # D = z_norm(D)
        print('shape = ', D.shape)
        print('shuffling data', file=sys.stderr)
        D, L = shuffle(D, L)

        
        Ls = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]#, 10]
        for l in Ls:
            print(f'lambda = {l}')
            model = models.LRBinClassifier(l, prior=0.1)
            scores = runKfold(D, L, 5, model)
            print('computing minDCF - actDCF')
            list_of_eff_priors = [0.1, 0.5]
            minDCFs, DCFs = compute_minDCF_optimized(list_of_eff_priors, scores, L), compute_actDCF(list_of_eff_priors, scores, L)
            print('minDCF = ' + str(minDCFs), file=sys.stderr)
            print('minDCF = ' + str(minDCFs))
            print('actDCF = ' + str(DCFs))

def save_scores(scores, L, file_name):
    path = 'scores/lr/uncal/'+file_name
    print(f'saving scores at {path}', file=sys.stderr)
    np.save(path, np.vstack([scores, L]))

def run():
    ms = [7]
    for m in ms:
        print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
        print('loading data', file=sys.stderr)
        D, L = load('data/Train.txt')
        print('center data', file=sys.stderr)
        if not m is None:
            D, _ = PCA(D, m)
        D = z_norm(D)
        print('shuffling data', file=sys.stderr)
        D, L = shuffle(D, L)

        D = expand_dataset(D)
        
        Ls = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        for l in Ls:
            print(f'lambda = {l}')
            model = models.LRBinClassifier(l, prior=0.1)
            scores = runKfold(D, L, 5, model)
            save_scores(scores, L, f'qlr_scores_{m}_z_1e{int(np.log10(l))}.npy')
            # print('computing minDCF - actDCF')
            # list_of_eff_priors = [0.1, 0.5]
            # minDCFs, DCFs = compute_minDCF_optimized(list_of_eff_priors, scores, L), compute_actDCF(list_of_eff_priors, scores, L)
            # print('minDCF = ' + str(minDCFs), file=sys.stderr)
            # print('minDCF = ' + str(minDCFs))
            # print('actDCF = ' + str(DCFs))
            # print()

def run_calibration():
    m = 7
    l = 1e-4
    print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
    print('loading data', file=sys.stderr)
    D, L = load('data/Train.txt')
    print('center data', file=sys.stderr)
    if not m is None:
        D, _ = PCA(D, m)
    D = z_norm(D)
    print('shuffling data', file=sys.stderr)
    D, L = shuffle(D, L)

    D = expand_dataset(D)
    
    print(f'lambda = {l}')
    model = models.LRBinClassifier(l, prior=0.1)
    scores = runKfold(D, L, 5, model)
    run_mindcf(scores, L)

    scores = vrow(scores)
    scores, L = shuffle(scores, L)
    cal_scores = runKfold_cal(scores, L, K=5)
    run_mindcf(cal_scores, L)
    print()
