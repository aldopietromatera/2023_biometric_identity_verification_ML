from utils import load, attribute_hists, shuffle, plot_correlations, split_db_2to1, center, plot_hist_scatter
from validation import Kfold
from measuring_predictions import compute_min_DCF
import numpy as np
from dimred import PCA, LDA
import utils
import preproc
import lr
import svm
import gmm
import mvg
import calibration
import evaluation

def main():
    print("--------- Started ---------")
    evaluation.eval_qlr_complete()
    print("--------- All finished ---------")

if __name__ == "__main__":
    main()



