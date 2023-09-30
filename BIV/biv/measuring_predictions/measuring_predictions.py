import numpy as np
import matplotlib.pyplot as plt

def conf_mat(L, Lpred):
    dim = np.max(L)+1
    ret = np.zeros([dim, dim], dtype=int)
    for i in range(len(L)):
        ret[Lpred[i], L[i]] += 1
    return ret


def get_confusion_matrix(num_classes: int, predicted_label, actual_label):
    """Return the confusion matrix

    Args:
        num_classes (int): number of classes
        predicted_label (_type_): array of the predicted labels
        actual_label (_type_): array of the actual labels

    Returns:
        np array of shape (num_classes, num_classes)
    """
    conf_matr = np.zeros((num_classes, num_classes))
    for p, a in zip(predicted_label, actual_label):
        conf_matr[int(p)][int(a)] += 1
    return conf_matr


def getFNR(M):
    return M[0, 1] / (M[0, 1] + M[1, 1])


def getFPR(M):
    return M[1, 0] / (M[1, 0] + M[0, 0])


def getTPR(M):
    return 1-getFNR(M)


def getTNR(M):
    return 1-getFPR(M)


def bayes_dummy(pi1, Cfn, Cfp):
    return min(pi1*Cfn, (1-pi1)*Cfp)


def bayes_risk(M, pi1, Cfn, Cfp):
    FNR = getFNR(M)
    FPR = getFPR(M)
    return pi1*Cfn*FNR + (1-pi1)*Cfp*FPR


def sort_llrs_with_labels(llrs, L):
    tmp = np.vstack([llrs, L]).T
    # tmp = np.sort(tmp, 0)
    idxes = np.argsort(tmp[:, 0])
    tmp = tmp[idxes]
    llrs = tmp[:, 0].reshape(llrs.shape)
    L = tmp[:, 1].reshape(L.shape).astype(int)
    return llrs, L


def plot_ROC_curve(FPR, TPR, title):
    plt.plot(FPR, TPR)
    tit = "ROC curve - {}".format(title)
    plt.title(tit)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


def bayes_error_plots_optimized(llrs, L, color1, color2, show=True, newfig=True, legend=['actDCF', 'minDCF']):
    llrs, L = sort_llrs_with_labels(llrs, L)
    effPriorLogOdds = np.linspace(-3, 3, 21)
    tresholds = np.array([-np.infty, *llrs, np.infty])
    DCFs = []
    minDCFs = []
    for eplo in effPriorLogOdds:
        effPrior = 1/(1+np.exp(-eplo))
        t = -eplo
        Lpred = (llrs > t).astype(int)
        M = conf_mat(L, Lpred)
        DCF = bayes_risk(M, effPrior, 1, 1)
        DCFnorm = DCF / bayes_dummy(effPrior, 1, 1)
        DCFs.append(DCFnorm)
        minDCF = None
        M = None
        for i in range(len(tresholds)):
            if M is None:
                M = conf_mat(L, np.ones(L.shape[0], dtype=int))
            else:
                if i-1 == L.shape[0]:
                    pass
                elif L[i-1] == 1:
                    M[0, 1] += 1
                    M[1, 1] -= 1
                else:
                    M[0, 0] += 1
                    M[1, 0] -= 1
            DCF = bayes_risk(M, effPrior, 1, 1)
            DCFnorm = DCF / bayes_dummy(effPrior, 1, 1)
            if minDCF is None or minDCF > DCFnorm:
                minDCF = DCFnorm
        minDCFs.append(minDCF)
    if newfig:
        plt.figure()
        plt.grid(True)
    plt.plot(effPriorLogOdds, DCFs, label='actDCF', color=color1, linestyle='solid')
    plt.plot(effPriorLogOdds, minDCFs, label='minDCF', color=color2, linestyle='dashed')
    if not legend is None:
        plt.legend()
    if show:
        plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")
        plt.ylabel("DCF")
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.show()


def det_plot(llrs, L, newfig=True, show=True, legend=None):
    llrs, L = sort_llrs_with_labels(llrs, L)
    tresholds = np.array([-np.infty, *llrs, np.infty])
    tmp = []
    M = None
    for i in range(len(tresholds)):
        if M is None:
            M = conf_mat(L, np.ones(L.shape[0], dtype=int))
        else:
            if i-1 == L.shape[0]:
                pass
            elif L[i-1] == 1:
                M[0, 1] += 1
                M[1, 1] -= 1
            else:
                M[0, 0] += 1
                M[1, 0] -= 1
        tmp.append([getFPR(M), getFNR(M)])
    tmp = np.array(tmp)
    # print(tmp)
    # tmp = np.sort(tmp, 0)
    # idxes = np.argsort(tmp[:, 0])
    # tmp = tmp[idxes]
    FPR = tmp[:, 0].T
    FNR = tmp[:, 1].T
    if newfig:
        plt.figure()
    plt.grid(True)
    plt.loglog(np.array(FPR), np.array(FNR), linewidth=2)
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    if not legend is None:
        plt.legend(legend)
    if show:
        plt.title('DET plot')
        plt.show()

    # plt.legend(['actDCF', 'minDCF'])
    # plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")
    # plt.ylabel("DCF")
    # plt.ylim([0, 1.1])
    # plt.xlim([-3, 3])
    

def compute_minDCF_optimized(eff_priors, llrs, L):
    effPriorLogOdds = np.linspace(-3, 3, 21)

    llrs, L = sort_llrs_with_labels(llrs, L)
    tresholds = np.array([-np.infty, *llrs])
    minDCFs = []
    for pi in eff_priors:
        minDCF = None
        M = None
        for i in range(len(tresholds)):
            t = tresholds[i]
            # print(f'treshold {t}')
            if M is None:
                M = conf_mat(L, np.ones(L.shape[0], dtype=int))
            else:
                if L[i-1] == 1:
                    M[0, 1] += 1
                    M[1, 1] -= 1
                else:
                    M[0, 0] += 1
                    M[1, 0] -= 1
            # print(f'TP = {M[1,1]} TN = {M[0,0]} FP = {M[1,0]} FN = {M[0,1]}')
            DCFu = bayes_risk(M, pi, 1, 1)
            DCFnorm = DCFu / bayes_dummy(pi, 1, 1)
            if minDCF == None or minDCF > DCFnorm:
                minDCF = DCFnorm
        minDCFs.append(minDCF)

    # printing
    # print("(pi, cfn, cfp) | DCFu | DCF | minDCF = ({0}, {1}, {2}) | {3:.3f} | {4:.3f}".format(pi_true, cfn, cfp, DCFu, DCFn))
    # print("(pi, cfn, cfp) | DCF | minDCF = ({0:.3f}, {1}, {2}) | {3:.3f} | {4:.3f}".format(pi, 1, 1, DCF_for_this_pi, minDCF))
    # plot_ROC_curve(FPRs, TPRs, "({0}, {1}, {2})".format(pi_true, cfn, cfp))
    # print("--------------------------------\n")
    # bayes error plots
    # plt.plot(effPriorLogOdds, DCFs_to_print, label="DCF", color="r")
    # plt.plot(effPriorLogOdds, minDCFs, label="min DCF", color="b")
    # plt.ylim([0, 1.1])
    # plt.xlim([-3, 3])
    # plt.legend()
    # plt.show()

    return minDCFs


def compute_actDCF(eff_priors, llrs, L):
    actDCFs = []
    for pi in eff_priors:
        t = -1*np.log(pi/(1-pi))
        Lpred = np.array(llrs > t)
        M = get_confusion_matrix(2, Lpred, L)
        DCFu = bayes_risk(M, pi, 1, 1)
        DCFnorm = DCFu/bayes_dummy(pi, 1, 1)
        actDCFs.append(DCFnorm)
    return actDCFs


def _compute_binary_optimal_bayes_decision(llrs, actual_labels, pi_true, cost_false_negative=1, cost_false_positive=1, t=None):
    """

    Args:
        llrs (_type_): log likelihood ratios
        actual_labels (_type_): actual labels
        pi_true (_type_): effective prior
        cost_false_negative (int, optional): Cfn. Defaults to 1.
        cost_false_positive (int, optional): Cfp. Defaults to 1.
        t (_type_, optional): threshold. Defaults to None.

    Returns:
        np array of shape (num_classes, num_classes) from get_confusion_matrix
    """
    if t is None:
        t = -1*np.log((pi_true*cost_false_negative) /
                      ((1-pi_true)*cost_false_positive))
    predicted_labels = np.array(llrs > t)
    conf_matr = get_confusion_matrix(2, predicted_labels, actual_labels)

    return conf_matr


def _compute_bayes_risk_binary(conf_matr, pi_true, cfn=1, cfp=1):
    """

    Args:
        conf_matr (_type_): confusion matrix
        pi_true (_type_): effective prior
        cfn (int, optional): Cfn. Defaults to 1.
        cfp (int, optional): Cfp. Defaults to 1.

    Returns:
        not normalized DCF
    """
    M = conf_matr
    FNR = M[0][1] / (M[0][1]+M[1][1])
    FPR = M[1][0] / (M[0][0]+M[1][0])
    # TPR = 1-FNR

    DCFu = pi_true*cfn*FNR + (1-pi_true)*cfp*FPR

    return DCFu


def _compute_normalized_DCF(DCFu, pi_true, cfn=1, cfp=1):
    """

    Args:
        DCFu (_type_): not normalized DCF
        pi_true (_type_): effective prior
        cfn (int, optional): Cfn. Defaults to 1.
        cfp (int, optional): Cfp. Defaults to 1.

    Returns:
        normalized DCF
    """
    B_dummy = min(pi_true*cfn, (1-pi_true)*cfp)
    return DCFu/B_dummy


def compute_min_DCF(list_of_eff_priors, llr, actual_labels):
    """

    Args:
        list_of_eff_priors (_type_): list of effective priors on which wew want to evaluate the minDCF
        llr (_type_): log likelihood ratis
        actual_labels (_type_): actual labels

    Returns:
        list of minDCFs for each input effective priors in list_of_eff_priors argument
        and 
        list of DCF for each input effective prior
    """
    test_scores_sorted = np.sort(llr)

    # Bayes error plots
    # effPriorLogOdds = np.linspace(-3, 3, 21)
    # effPrior = 1/(1+np.exp(-effPriorLogOdds))

    set_of_t = [-np.inf]
    for a in test_scores_sorted:
        set_of_t.append(a)
    set_of_t.append(np.inf)
    set_of_t.append(None)

    minDCFs = []
    DCFs_to_print = []
    for pi in list_of_eff_priors:
        DCFs = []
        for t in set_of_t:
            conf_matr = _compute_binary_optimal_bayes_decision(
                llr, actual_labels, pi, t=t)
            DCFu = _compute_bayes_risk_binary(conf_matr, pi)
            DCFn = _compute_normalized_DCF(DCFu, pi)

            DCFs.append(DCFn)

        DCF_for_this_pi = DCFs.pop()
        DCFs_to_print.append(DCF_for_this_pi)
        minDCF = min(DCFs)
        # print("pi= {} - minDCF= {}, DCF= {}".format(pi, minDCF, DCF_for_this_pi))
        minDCFs.append(minDCF)

    return minDCFs, DCFs_to_print

    # printing
    # print("(pi, cfn, cfp) | DCFu | DCF | minDCF = ({0}, {1}, {2}) | {3:.3f} | {4:.3f}".format(pi_true, cfn, cfp, DCFu, DCFn))
    # print("(pi, cfn, cfp) | DCF | minDCF = ({0:.3f}, {1}, {2}) | {3:.3f} | {4:.3f}".format(pi, 1, 1, DCF_for_this_pi, minDCF))
    # plot_ROC_curve(FPRs, TPRs, "({0}, {1}, {2})".format(pi_true, cfn, cfp))
    # print("--------------------------------\n")
    # bayes error plots
    # plt.plot(effPriorLogOdds, DCFs_to_print, label="DCF", color="r")
    # plt.plot(effPriorLogOdds, minDCFs, label="min DCF", color="b")
    # plt.ylim([0, 1.1])
    # plt.xlim([-3, 3])
    # plt.legend()
    # plt.show()
