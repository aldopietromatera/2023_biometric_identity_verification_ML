from utils import load, shuffle, z_norm, cartesian_to_polar, vrow
import models
from validation import Kfold
from measuring_predictions import compute_min_DCF, compute_minDCF_optimized, compute_actDCF
import numpy as np
from dimred import PCA
import sys


def runKfold(D, L, K, model):
    print('running K-fold', file=sys.stderr)
    LLRs = []

    def kfold_operation(cnt, DTR, LTR, DTE, LTE):
        print(f'running k-fold on fold #{cnt}', file=sys.stderr)
        model.train(DTR, LTR)
        _, llrs = model.test(DTE)
        LLRs.append(llrs)

    Kfold(D, L, K, kfold_operation)
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


def run_old():
    ms = [None, 10, 9, 8, 7, 6]
    # ms = [None]
    for m in ms:
        print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
        print('loading data')
        D, L = load('data/Train.txt')
        if not m is None:
            D, _ = PCA(D, m)

        D = z_norm(D)
        print('shuffling data\n')
        D, L = shuffle(D, L)

        d = 2
        Cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        Ks = [0.1, 1, 10, 100]
        gammas = [0.001, 0.01, 0.1, 1]
        # Cs = [1e-4]
        # Ks = [50]
        # gammas = []

        print('------poly kernel------')
        for C in Cs:
            for K in Ks:
                model = models.NL_SVM(C, K, models.Kernel.poly(d, 1))
                print(f'd={d} C={C} K={K}')
                scores = runKfold(D, L, K=5, model=model)
                run_mindcf(scores, L)
                print()

        print('------RBF kernel------')
        for C in Cs:
            for K in Ks:
                for g in gammas:
                    model = models.NL_SVM(C, K, models.Kernel.RBF(g))
                    print(f'C={C} K={K} gamma={g}')
                    scores = runKfold(D, L, K=5, model=model)
                    run_mindcf(scores, L)
                    print()

def run_calibration():
    m = 7
    g = 0.16
    C = 1
    K = 0.1
    print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
    print('loading data', file=sys.stderr)
    D, L = load('data/Train.txt')
    if not m is None:
        D, _ = PCA(D, m)
    D = z_norm(D)
    print('shuffling data\n', file=sys.stderr)
    D, L = shuffle(D, L)
    
    print('------RBF kernel------')
    model = models.NL_SVM(C, K, models.Kernel.RBF(g))
    print(f'C={C} K={K} gamma={g}')
                    
    
    scores = runKfold(D, L, K=5, model=model)
    run_mindcf(scores, L)
    
    scores = vrow(scores)
    scores, L = shuffle(scores, L)
    cal_scores = runKfold_cal(scores, L, K=5)
    run_mindcf(cal_scores, L)
    print()

def save_scores(scores, L, file_name):
    path = 'scores/svm/uncal/'+file_name
    print(f'saving scores at {path}', file=sys.stderr)
    np.save(path, np.vstack([scores, L]))

def run():
    ms = [8]
    for m in ms:
        print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
        print('loading data', file=sys.stderr)
        D, L = load('data/Train.txt')
        if not m is None:
            D, _ = PCA(D, m)
        D = z_norm(D)
        print('shuffling data\n', file=sys.stderr)
        D, L = shuffle(D, L)

        d = 2
        Cs = [1e-3, 1e-2, 1e-1, 1]
        Ks = [0.1, 1, 10, 100]
        # gammas = [0.001, 0.01, 0.1, 1]
        # Cs = [1]
        # Ks = [0.1]
        # gammas = np.linspace(0.01, 0.2, 20)
        # print(gammas, file=sys.stderr)
        
        print('------poly kernel------')
        for C in Cs:
            for K in Ks:
                model = models.NL_SVM(C, K, models.Kernel.poly(d, 1))
                print(f'C={C} K={K}')
                scores = runKfold(D, L, K=5, model=model)
                save_scores(scores, L, f'polysvm_scores_{m}_z_1e{int(np.log10(C))}_1e{int(np.log10(K))}_{d}.npy')
                # run_mindcf(scores, L)
                # print()


def run_linear():
    ms = [None]
    for m in ms:
        print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
        print('loading data', file=sys.stderr)
        D, L = load('data/Train.txt')
        if not m is None:
            D, _ = PCA(D, m)
        D = cartesian_to_polar(D)
        print(D.shape)
        # D = z_norm(D)
        print('shuffling data\n', file=sys.stderr)
        D, L = shuffle(D, L)

        Cs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        Ks = [0.1, 1, 10, 100]
        
        print('------NO kernel------')
        for C in Cs:
            for K in Ks:
                model = models.SVM(C, K)
                print(f'C={C} K={K}')
                scores = runKfold(D, L, K=5, model=model)
                run_mindcf(scores, L)
                print()