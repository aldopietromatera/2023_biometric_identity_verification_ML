from utils import load, attribute_hists, shuffle, plot_correlations, split_db_2to1, center, plot_hist_scatter, z_norm
import models
from validation import Kfold
from measuring_predictions import compute_min_DCF
import numpy as np
from dimred import PCA
import sys


def runKfold(D, L, K, Kcs, diag, tied):
    LLRs = []

    def kfold_operation(cnt, DTR, LTR, DTE, LTE):
        print(f'running k-fold on fold #{cnt}', file=sys.stderr)
        model = models.GMMClassifier(diagonal=diag, tied=tied)
        model.train(DTR, LTR, Kcs)
        _, llrs = model.test(DTE)
        LLRs.append(llrs)

    Kfold(D, L, K, kfold_operation)
    return np.array(LLRs).reshape(L.shape)


def run_nb():
    ms = [10, 9, 8, 7]
    # ms = [None]
    K = [2]
    diags = [False]
    tieds = [False]
    for diag in diags:
        for tied in tieds:
            print(f'diag = {diag} tied = {tied}')
            for m in ms:
                print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
                for k1 in K:
                    #for k2 in K:
                    k2 = k1
                    print(f'K0={k1} K1={k2}')
                    print(f'K0={k1} K1={k2}', file=sys.stderr)
                    print('loading data', file=sys.stderr)
                    D, L = load('data/Train.txt')
                    if not m is None:
                        D, _ = PCA(D, m)
                    # D = z_norm(D)
                    
                    print('shuffling data', file=sys.stderr)
                    D, L = shuffle(D, L)

                    print('run K-fold', file=sys.stderr)
                    LLRs = runKfold(D, L, K=5, Kcs=[k1, k2], diag=diag, tied=tied)
                    
                    print('computing minDCF - actDCF')
                    list_of_eff_priors = [0.1]#, 0.5]
                    minDCFs, DCFs = compute_min_DCF(list_of_eff_priors, LLRs, L)
                    print('minDCF = ' + str(minDCFs), file=sys.stderr)
                    print('minDCF = ' + str(minDCFs))
                    print('actDCF = ' + str(DCFs))
                    print()
                print()
            print()

def run():
    ms = [None, 10, 9, 8, 7]
    K = [1, 2, 4, 8, 16]
    diags = [False, True]
    tieds = [False, True]
    for diag in diags:
        for tied in tieds:
            print(f'diag = {diag} tied = {tied}')
            for m in ms:
                print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
                for k1 in K:
                    for k2 in K:
                        print(f'K0={k1} K1={k2}')
                        print(f'K0={k1} K1={k2}', file=sys.stderr)
                        print('loading data', file=sys.stderr)
                        D, L = load('data/Train.txt')
                        if not m is None:
                            D, _ = PCA(D, m)
                        # D = z_norm(D)
                        
                        print('shuffling data', file=sys.stderr)
                        D, L = shuffle(D, L)

                        print('run K-fold', file=sys.stderr)
                        LLRs = runKfold(D, L, K=5, Kcs=[k1, k2], diag=diag, tied=tied)
                        
                        print('computing minDCF - actDCF')
                        list_of_eff_priors = [0.1, 0.5]
                        minDCFs, DCFs = compute_min_DCF(list_of_eff_priors, LLRs, L)
                        print('minDCF = ' + str(minDCFs), file=sys.stderr)
                        print('minDCF = ' + str(minDCFs))
                        print('actDCF = ' + str(DCFs))
                        print()
                print()
            print()
