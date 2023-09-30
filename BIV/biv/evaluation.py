from utils import load, shuffle, z_norm, vrow, split_db_2to1, z_norm_train_test, expand
import models
from validation import Kfold
from measuring_predictions import compute_minDCF_optimized, compute_actDCF, bayes_error_plots_optimized, det_plot
import numpy as np
from dimred import PCA, PCA_train_test
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


def evaluate_gmm():
    print('Evaluating GMM Tied K=[2,2] + NO PCA')
    DTR, LTR = load('data/Train.txt')
    DTE, LTE = load('data/Test.txt')
    model = models.GMMClassifier(tied=True, diagonal=False)
    model.train(DTR, LTR, [2, 2])
    _, scores = model.test(DTE)
    scores = np.array(scores)
    run_mindcf(scores, LTE)
    print()


def evaluate_mvg():
    print('Evaluating MVG Naive + PCA(8)')
    DTR, LTR = load('data/Train.txt')
    DTE, LTE = load('data/Test.txt')
    DTR, DTE = PCA_train_test(DTR, DTE, 8)
    model = models.MVGBinClassifier(naive_bayes=True, tied=False)
    model.train(DTR, LTR)
    _, scores = model.test(DTE)
    scores = np.array(scores)
    run_mindcf(scores, LTE)
    print()


def evaluate_svmpoly():
    print('Evaluating SVM poly d=2, C=0.01, K=10 + PCA(8) + z-norm + calibration (pi_T=0.1)')
    DTR, LTR = load('data/Train.txt')
    DTE, LTE = load('data/Test.txt')

    DTR, DTE = PCA_train_test(DTR, DTE, 8)
    DTR, DTE = z_norm_train_test(DTR, DTE)

    model = models.NL_SVM(C=0.01, K=10, kernel=models.Kernel.poly(2, 1))
    DTRc, LTRc = load_scores(SVM_POLY_FILE)
    cal_model = models.LRBinClassifier(l=0, prior=0.1)

    model.train(DTR, LTR)
    cal_model.train(vrow(DTRc), LTRc)

    _, scores = model.test(DTE)
    scores = vrow(np.array(scores))
    _, cal_scores = cal_model.test(scores)
    cal_scores -= np.log(0.1/0.9)
    run_mindcf(cal_scores, LTE)
    print()


def evaluate_svmrbf():
    print('Evaluating SVM RBF C=1, K=0.1, g=0.16 + PCA(7) + z-norm + calibration (pi_T=0.1)')
    DTR, LTR = load('data/Train.txt')
    DTE, LTE = load('data/Test.txt')

    DTR, DTE = PCA_train_test(DTR, DTE, 7)
    DTR, DTE = z_norm_train_test(DTR, DTE)

    model = models.NL_SVM(C=1, K=0.1, kernel=models.Kernel.RBF(0.16))
    DTRc, LTRc = load_scores(SVM_RBF_FILE)
    cal_model = models.LRBinClassifier(l=0, prior=0.1)

    model.train(DTR, LTR)
    cal_model.train(vrow(DTRc), LTRc)

    _, scores = model.test(DTE)
    scores = vrow(np.array(scores))
    _, cal_scores = cal_model.test(scores)
    cal_scores -= np.log(0.1/0.9)
    run_mindcf(cal_scores, LTE)
    print()


def eval_svm_poly_complete():
    print('Complete Evaluation of SVM poly d=2 + PCA(8) + z-norm + calibration (pi_T=0.1)')
    DTR, LTR = load('data/Train.txt')
    DTE, LTE = load('data/Test.txt')

    DTR, DTE = PCA_train_test(DTR, DTE, 8)
    DTR, DTE = z_norm_train_test(DTR, DTE)

    d = 2
    Cs = [1e-3, 1e-2, 1e-1, 1]
    Ks = [0.1, 1, 10, 100]
    m = 8
    for C in Cs:
        for K in Ks:
            print(f'C={C} K={K}')
            DTRc, LTRc = load_scores(
                f'scores/svm/uncal/polysvm_scores_{m}_z_1e{int(np.log10(C))}_1e{int(np.log10(K))}_{d}.npy')
            DTRc = vrow(DTRc)

            model = models.NL_SVM(C, K, models.Kernel.poly(d, 1))
            cmodel = models.LRBinClassifier(0, 0.1)

            model.train(DTR, LTR)
            cmodel.train(DTRc, LTRc)

            _, scores = model.test(DTE)
            scores = vrow(np.array(scores))
            _, cscores = cmodel.test(scores)
            cscores -= np.log(1/9)
            run_mindcf(cscores, LTE)
            print()


def eval_svm_rbf_complete():
    print('Complete Evaluation of SVM RBF + PCA(7) + z-norm + calibration (pi_T=0.1)')
    DTR, LTR = load('data/Train.txt')
    DTE, LTE = load('data/Test.txt')

    DTR, DTE = PCA_train_test(DTR, DTE, 7)
    DTR, DTE = z_norm_train_test(DTR, DTE)

    gammas = [0.001, 0.01, 0.1, 1]
    Cs = [1e-1, 1]
    Ks = [0.1, 1, 10, 100]
    m = 7
    for C in Cs:
        for K in Ks:
            for g in gammas:
                print(f'C={C} K={K}')
                DTRc, LTRc = load_scores(
                    f'scores/svm/uncal/rbfsvm_scores_{m}_z_1e{int(np.log10(C))}_1e{int(np.log10(K))}_1e{int(np.log10(g))}.npy')
                DTRc = vrow(DTRc)

                model = models.NL_SVM(C, K, models.Kernel.RBF(g))
                cmodel = models.LRBinClassifier(0, 0.1)

                model.train(DTR, LTR)
                cmodel.train(DTRc, LTRc)

                _, scores = model.test(DTE)
                scores = vrow(np.array(scores))
                _, cscores = cmodel.test(scores)
                cscores -= np.log(1/9)
                run_mindcf(cscores, LTE)
                print()


def expand_dataset(D):
    D2 = []
    # print(D.shape)
    for i in range(D.shape[1]):
        D2.append(expand(D[:, i]))
    # print(np.hstack(D2).shape)
    return np.hstack(D2)


def eval_qlr_complete():
    print('Complete Evaluation of QLR + PCA(7) + z-norm + calibration (pi_T=0.1)')
    DTR, LTR = load('data/Train.txt')
    DTE, LTE = load('data/Test.txt')

    DTR, DTE = PCA_train_test(DTR, DTE, 8)
    DTR, DTE = z_norm_train_test(DTR, DTE)
    DTR = expand_dataset(DTR)
    DTE = expand_dataset(DTE)

    m = 7
    Ls = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    for l in Ls:
        print(f'lambda = {l}')
        DTRc, LTRc = load_scores(
            f'scores/lr/uncal/qlr_scores_{m}_z_1e{int(np.log10(l))}.npy')
        DTRc = vrow(DTRc)

        model = models.LRBinClassifier(l, prior=0.1)
        cmodel = models.LRBinClassifier(0, prior=0.1)

        model.train(DTR, LTR)
        cmodel.train(DTRc, LTRc)

        _, scores = model.test(DTE)
        scores = vrow(np.array(scores))
        _, cscores = cmodel.test(scores)
        cscores -= np.log(1/9)
        run_mindcf(cscores, LTE)
        print()


def eval_mvg_complete():
    print('Complete Evaluation of MVG (standard / naive)')
    ms = [None, 10, 9, 8, 7, 6]
    for m in ms:
        DTR, LTR = load('data/Train.txt')
        DTE, LTE = load('data/Test.txt')
        for naive in [False, True]:
            print(
                f'PCA({"None" if m is None else m}) {"naive" if naive else "standard"}')
            if m != None:
                DTR, DTE = PCA_train_test(DTR, DTE, m)
            model = models.MVGBinClassifier(naive_bayes=naive, tied=False)
            model.train(DTR, LTR)
            _, scores = model.test(DTE)
            run_mindcf(scores, LTE)
            print()


def eval_gmm_complete():
    ms = [None, 8]
    K = [1, 2, 4, 8, 16]
    tieds = [False, True]
    for m in ms:
        DTR, LTR = load('data/Train.txt')
        DTE, LTE = load('data/Test.txt')
        for tied in tieds:
            for k1 in K:
                for k2 in K:
                    print(
                        f'PCA({"None" if m is None else m}) {"tied" if tied else "full cov."} K=[{k1} {k2}]')
                    if m != None:
                        DTR, DTE = PCA_train_test(DTR, DTE, m)
                    model = models.GMMClassifier(tied=tied, diagonal=False)
                    model.train(DTR, LTR, [k1, k2])
                    _, scores = model.test(DTE)
                    run_mindcf(scores, LTE)
                    print()
