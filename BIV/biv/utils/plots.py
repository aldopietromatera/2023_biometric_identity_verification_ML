import matplotlib.pyplot as plt
import numpy
import seaborn as sns

from utils import compute_correlation

RED = (0.7, 0.3, 0.1, 0.5)
BLUE = (0.3, 0.2, 0.7, 0.5)
GREEN = (0.2, 0.7, 0.3, 0.5)
ORANGE = (0.6, 0.5, 0.2, 0.5)

colors = [RED, BLUE]
def plot_attribute(dataset, datalabels, attribute_idx, legend=['0', '1'], savedir=None, subplot = None):
    """_summary_
        (works for binary problems)
        plots histograms of an attribute for each class
    Args:
        dataset (_type_): dataset to plot
        datalabels (_type_): labels corresponding to dataset samples
        attribute_idx (_type_): index of the attribute to plot
        legend (_type_): legend to visualize on the plot. Defaults to ['0', '1'].
    """
    if subplot is None:
        plt.figure()
        plt.hist(dataset[attribute_idx, datalabels==0], color=colors[0], density=True, bins=50, label=legend[0])
        plt.hist(dataset[attribute_idx, datalabels==1], color=colors[1], density=True, bins=50, label=legend[1])
        plt.legend()
        plt.title('attribute ' + str(attribute_idx))
        if savedir is None:
            plt.show()
        else:
            plt.savefig(savedir+'/attr_'+str(attribute_idx))
    else:
        subplot.hist(dataset[attribute_idx, datalabels==0], color=colors[0], density=True, bins=50, label=legend[0], alpha = 0.5)
        subplot.hist(dataset[attribute_idx, datalabels==1], color=colors[1], density=True, bins=50, label=legend[1], alpha = 0.5)
        subplot.legend()
        subplot.set_title('attribute ' + str(attribute_idx))


def attribute_hists(D, L, rows, cols, file_name, legend=['0', '1'], figsize=None):
    num_features = D.shape[0]
    assert(rows*cols == num_features)
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    # print(axs)
    for i in range(rows):
        for j in range(cols):
            # print(axs[i,j])
            plot_attribute(D, L, i+j*rows, legend, subplot=axs[i, j])
    plt.tight_layout()
    plt.savefig(file_name)

def plot_couple_of_attributes(dataset, datalabels, attr1_idx, attr2_idx, legend = ['0', '1'], subplot=None):
    """_summary_
        (works for binary problems)
        plots a scatter plot of the labels distribution wrt a couple of attributes for each class
    Args:
        dataset (_type_): dataset to plot
        datalabels (_type_): labels corresponding to dataset samples
        attr1_idx (_type_): index of the first attribute
        attr2_idx (_type_): index of the second attribute
        legend (_type_): legend to visualize on the plot. Defaults to ['0', '1'].
    """
    if subplot is None:
        plt.figure()
        plt.scatter(dataset[attr1_idx, datalabels==0], dataset[attr2_idx, datalabels==0], color=[colors[0]])
        plt.scatter(dataset[attr1_idx, datalabels==1], dataset[attr2_idx, datalabels==1], color=[colors[1]])
        plt.legend(legend)
        plt.xlabel('attribute ' + str(attr1_idx))
        plt.ylabel('attribute ' + str(attr2_idx))
        plt.savefig("./images/scatterplots/attr_"+str(attr1_idx)+"_attr_"+str(attr2_idx))
    else:
        subplot.scatter(dataset[attr1_idx, datalabels==0], dataset[attr2_idx, datalabels==0], color=[colors[0]], alpha = 0.5)
        subplot.scatter(dataset[attr1_idx, datalabels==1], dataset[attr2_idx, datalabels==1], color=[colors[1]], alpha = 0.5)

def plot_hist_scatter(dataset, datalabels, rows, cols, file_name, figsize=(25, 25)):
    """plot a unique figure in which there are hist and scatterplots

    Args:
        dataset (_type_): dataset
        datalabels (_type_): datalabels
        rows (_type_): number of rows of the figure
        cols (_type_): number of cols of the figure
        file_name (_type_): path where to save the figure
        figsize (tuple, optional): size of each small figure in the subplot. Defaults to (25, 25).
    """    
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if i == j:
                plot_attribute(dataset, datalabels, i, subplot=axs[i, j])
            else:
                plot_couple_of_attributes(dataset, datalabels, i, j, subplot=axs[i, j])

    plt.savefig(file_name)

def plot_2D_graph(X, Y, labels=['x','y'], file_name=None):
    plt.figure()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.grid(visible=True)
    plt.plot(X, Y, marker='o')
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)


def plot_correlations(DTR, title, graph_title=None, cmap="Greys"):
    n = DTR.shape[0]
    corr = numpy.zeros((n, n))
    for x in range(n):
        for y in range(n):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem
    sns.set()
    
    ax = plt.axes()
    heatmap = sns.heatmap(numpy.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False, ax = ax)
    if not graph_title is None:
        ax.set_title(graph_title)
    plt.show()
    fig = heatmap.get_figure()
    fig.savefig("./images/heatmaps/" + title + ".png")