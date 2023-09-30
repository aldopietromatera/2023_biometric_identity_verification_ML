from utils import load, shuffle, z_norm, cartesian_to_polar, vrow
import models
from validation import Kfold
from measuring_predictions import compute_minDCF_optimized, compute_actDCF, bayes_error_plots_optimized, det_plot
import numpy as np
from dimred import PCA
import sys
import latex


def runKfold(D, L, K, model, Kcs=None):
    print('running K-fold', file=sys.stderr)
    LLRs = []

    def kfold_operation(cnt, DTR, LTR, DTE, LTE):
        print(f'running k-fold on fold #{cnt}', file=sys.stderr)
        model.train(DTR, LTR, Kcs)
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
    minDCFs, DCFs = compute_minDCF_optimized(
        list_of_eff_priors, scores, L), compute_actDCF(list_of_eff_priors, scores, L)
    print('list_of_eff_priors = ' + str(list_of_eff_priors))
    # print('minDCF = ' + str(minDCFs), file=sys.stderr)
    print('minDCF = ' + str(minDCFs))
    print('actDCF = ' + str(DCFs))


def save_scores(file_name, scores, L):
    print(f'saving on {file_name}', file=sys.stderr)
    np.save(file_name, np.vstack([scores, L]))


def load_scores(file_name):
    print(f'loading from {file_name}', file=sys.stderr)
    data = np.load(file_name)
    scores = data[0]
    L = data[1]
    return scores, L


SVM_POLY_FILE = 'scores/svm_poly.npy'
SVM_RBF_FILE = 'scores/svm_rbf.npy'
GMM_TIED_FILE = 'scores/gmm_tied.npy'
MVG_NAIVE_FILE = 'scores/mvg_naive.npy'


def scores_svm_poly():
    m = 8
    d = 2
    C = 0.01
    K = 10
    print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
    print('loading data', file=sys.stderr)
    D, L = load('data/Train.txt')
    if not m is None:
        D, _ = PCA(D, m)
    D = z_norm(D)
    print('shuffling data\n', file=sys.stderr)
    D, L = shuffle(D, L)

    print('------poly kernel------')
    model = models.NL_SVM(C, K, models.Kernel.poly(d, 1))
    print(f'C={C} K={K} d={d}')

    scores = runKfold(D, L, K=5, model=model)
    # save_scores(SVM_POLY_FILE, scores, L)
    return scores, L


def scores_svm_rbf():
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
    save_scores(SVM_RBF_FILE, scores, L)
    return scores, L


def scores_gmm():
    m = None
    k1 = 2
    k2 = 2
    diag = False
    tied = True
    print(f'diag = {diag} tied = {tied}')
    print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
    print(f'K0={k1} K1={k2}')
    print(f'K0={k1} K1={k2}', file=sys.stderr)
    print('loading data', file=sys.stderr)
    D, L = load('data/Train.txt')
    if not m is None:
        D, _ = PCA(D, m)

    print('shuffling data', file=sys.stderr)
    D, L = shuffle(D, L)

    print('run K-fold', file=sys.stderr)
    model = models.GMMClassifier(tied=tied, diagonal=diag)
    LLRs = runKfold(D, L, K=5, model=model, Kcs=[k1, k2])
    save_scores(GMM_TIED_FILE, LLRs, L)
    return LLRs, L

def scores_mvg():
    m = 8
    naive = True
    tied = False
    print(f'naive = {naive} tied = {tied}')
    print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
    print('loading data', file=sys.stderr)
    D, L = load('data/Train.txt')
    if not m is None:
        D, _ = PCA(D, m)

    print('shuffling data', file=sys.stderr)
    D, L = shuffle(D, L)

    print('run K-fold', file=sys.stderr)
    model = models.MVGBinClassifier(naive_bayes=naive, tied=tied)
    LLRs = runKfold(D, L, K=5, model=model)
    save_scores(MVG_NAIVE_FILE, LLRs, L)
    return LLRs, L

def scores_gmm():
    m = None
    k1 = 2
    k2 = 2
    diag = False
    tied = True
    print(f'diag = {diag} tied = {tied}')
    print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
    print(f'K0={k1} K1={k2}')
    print(f'K0={k1} K1={k2}', file=sys.stderr)
    print('loading data', file=sys.stderr)
    D, L = load('data/Train.txt')
    if not m is None:
        D, _ = PCA(D, m)

    print('shuffling data', file=sys.stderr)
    D, L = shuffle(D, L)

    print('run K-fold', file=sys.stderr)
    model = models.GMMClassifier(tied=tied, diagonal=diag)
    LLRs = runKfold(D, L, K=5, model=model, Kcs=[k1, k2])
    save_scores(GMM_TIED_FILE, LLRs, L)
    return LLRs, L

def scores_gmm_naive():
    m = 8
    k1 = 4
    k2 = 4
    diag = False
    tied = True
    print(f'diag = {diag} tied = {tied}')
    print(f'>>>>    PCA({"None" if m is None else m})    <<<<')
    print(f'K0={k1} K1={k2}')
    print(f'K0={k1} K1={k2}', file=sys.stderr)
    print('loading data', file=sys.stderr)
    D, L = load('data/Train.txt')
    if not m is None:
        D, _ = PCA(D, m)

    print('shuffling data', file=sys.stderr)
    D, L = shuffle(D, L)

    print('run K-fold', file=sys.stderr)
    model = models.NaiveGMMClassifier(tied=tied, diagonal=diag)
    LLRs = runKfold(D, L, K=5, model=model, Kcs=[k1, k2])
    save_scores(GMM_TIED_FILE, LLRs, L)
    return LLRs, L

def run_calibration():
    # scores, L = scores_gmm()
    scores, L = load_scores(SVM_RBF_FILE)
    # print(scores)
    run_mindcf(scores, L)
    bayes_error_plots_optimized(scores, L, 'blue', 'mediumblue')
    scores = vrow(scores)
    scores, L = shuffle(scores, L)
    cal_scores = runKfold_cal(scores, L, K=5)
    run_mindcf(cal_scores, L)
    bayes_error_plots_optimized(cal_scores, L, 'green', 'darkgreen')


def run_fusion():
    models = [MVG_NAIVE_FILE, GMM_TIED_FILE, SVM_POLY_FILE, SVM_RBF_FILE]
    names = ['MVG naive', 'GMM tied', 'SVM poly', 'SVM rbf']
    combinations = [(0,1,2,3), (0,1,2,None), (0,1,3,None), (0,2,3,None), (1,2,3,None),
                    (0,1,None,None), (0,2,None,None), (0,3,None,None), (1,2,None,None),
                    (1,3,None,None), (2,3,None,None)]
    for i,j,k,w in combinations:
        print(f'{"" if i is None else names[i]} {"" if j is None else names[j]} {"" if k is None else names[k]} {"" if w is None else names[w]}')
        scores = []
        L = None
        if not i is None:
            s, L = load_scores(models[i])
            scores.append(s)
        if not j  is None:
            s, L = load_scores(models[j])
            scores.append(s)
        if not k  is None:
            s, L = load_scores(models[k])
            scores.append(s)
        if not w  is None:
            s, L = load_scores(models[w])
            scores.append(s)
        scores = np.vstack(scores)
        scores, L = shuffle(scores, L)
        cal_scores = runKfold_cal(scores, L, K=5)
        run_mindcf(cal_scores, L)

def run_det_plot():
    scores, L = load_scores(MVG_NAIVE_FILE)
    # scores, L = shuffle(vrow(scores), L)
    # cal_scores = runKfold_cal(scores, L, K=5)
    det_plot(scores, L, newfig=True, show=False)
    # bayes_error_plots_optimized(scores, L, 'blue', 'darkblue', False, True, None)

    scores, L = load_scores(SVM_POLY_FILE)
    scores, L = shuffle(vrow(scores), L)
    cal_scores = runKfold_cal(scores, L, K=5)
    det_plot(cal_scores, L, newfig=False, show=False)
    # bayes_error_plots_optimized(cal_scores, L, 'orange', 'darkorange', False, False, None)

    scores, L = load_scores(SVM_RBF_FILE)
    scores, L = shuffle(vrow(scores), L)
    cal_scores = runKfold_cal(scores, L, K=5)
    det_plot(cal_scores, L, newfig=False, show=False)
    # bayes_error_plots_optimized(cal_scores, L, 'green', 'darkgreen', False, False, None)

    scores, L = load_scores(GMM_TIED_FILE)
    # scores, L = shuffle(vrow(scores), L)
    # cal_scores = runKfold_cal(scores, L, K=5)
    det_plot(scores, L, newfig=False, show=True, legend=['MVG Naive + PCA(8)', 'SVM poly d=2, C=0.01, K=10 + PCA(8) + z-norm', 'SVM RBF C=1, K=0.1, $\\gamma=0.16$ + PCA(7) + z-norm', 'GMM Tied K0=K1=2'])
    # bayes_error_plots_optimized(scores, L, 'red', 'darkred', True, False, ['MVG Naive + PCA(8)', 'SVM poly d=2, C=0.01, K=10 + PCA(8) + z-norm', 'SVM RBF C=1, K=0.1, $\\gamma=0.16$ + PCA(7) + z-norm', 'GMM Tied K0=K1=2'])