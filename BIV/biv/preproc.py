import numpy as np
from dimred import PCA, LDA
from utils import load, plot_2D_graph, plot_correlations, plot_hist_scatter, plot_attribute


def run():
    plot_features()

def plot_lda_dir():
    print('loading data')
    D, L = load('data/Train.txt')
    D = LDA(D, L, 1)
    plot_attribute(D, L, 0, ['different speaker', 'same speaker'])

def plot_features():
    print('loading data')
    D, L = load('data/Train.txt')
    D, _ = PCA(D, 2, True)
    plot_hist_scatter(D, L, 2, 2, 'prova2.png')
    #plot_correlations(D, 'aaa', 'bbb')

def plot_retained_variance():
    print('loading data')
    D, L = load('data/Train.txt')
    _, s = PCA(D, 10, True)  # dummy call to get eigenvalues
    percs = np.cumsum(s) / np.sum(s)  # percentage of retained data
    percs = np.insert(percs, 0, 0)
    ndims = np.arange(11)
    print(ndims)
    print(percs)
    plot_2D_graph(ndims, percs, [
                  'dimensions', 'retained variance'], './images/preproc/retained_variance.png')


def plot_heatmaps():
    print('loading data')
    D, L = load('data/Train.txt')
    plot_correlations(D[:], 'class_both', 'entire dataset')
    plot_correlations(D[:, L == 1], 'class_1', 'same speaker')
    plot_correlations(D[:, L == 0], 'class_0', 'different speaker')

def get_dataset_prior():
    print('loading data')
    D, L = load('data/Train.txt')
    print((L==1).sum())
    print((L==0).sum())
