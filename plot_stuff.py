import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rc
import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import pandas as pd
import glob
import scipy.stats

# Add the module to source path
dirname = os.path.dirname(os.path.realpath(__file__))
source_path = dirname + "/portfolio"
sys.path.append(source_path)

from portfolio.plotting import plot_comparison

def isin(array, substring1, substring2=None):
    """
    Finds entries in array where substring1 is present, while substring2 is NOT present.
    https://stackoverflow.com/a/38974252/2653663
    """
    if substring2 is None:
        return np.flatnonzero(np.core.defchararray.find(array,substring1)!=-1)

    return np.flatnonzero((np.core.defchararray.find(array,substring1)!=-1) & (np.core.defchararray.find(array,substring2)==-1))

def load_pickle(filename):
    with open(filename, "rb") as f:
        d = pickle.load(f)
    return d

# set matplotlib defaults
sns.set(font_scale=1.0)
sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})
rc('text', usetex=False)

def get_best_params(params):
    """
    Attempts to get the best set of parameters
    from a list of dictionaries containing the
    parameters that resulted in the best cv score
    """

    # Preprocess a bit
    d = {}
    for param in params:
        for key,value in param.items():
            if key not in d: d[key] = []
            d[key].append(value)

    # Select the most likely or median value
    for key,value in d.items():
        # if list choose most common
        if isinstance(value[0], str):
            best_value = max(set(value), key=value.count)
            d[key] = best_value
            continue

        # if numeric return median
        d[key] = np.median(value)

    return d

def get_mean_lower_and_upper_bound(x, alpha=0.95, bootstrap=False):
    """
    Calculate 95% confidence interval for rapported MAE.
    The data is assumed to follow a laplacian distribution.
    See https://waset.org/publications/8809/confidence-intervals-for-double-exponential-distribution-a-simulation-approach
    for derivation.
    """
    # Number of datapoints
    m, n = x.shape
    if bootstrap:
        lb, ub = np.zeros(m), np.zeros(m)
        for i in range(m):
            X = np.random.choice(x[i], size=(n,10000))
            mae = np.mean(abs(X), axis=0)
            lb[i] = np.percentile(mae, 100-alpha*100/2)
            ub[i] = np.percentile(mae, alpha*100/2)

    else:
        lb = 2 * mae * n / scipy.stats.chi2.ppf((1 + alpha) / 2, 2 * n)
        ub = 2 * mae * n / scipy.stats.chi2.ppf((1 - alpha) / 2, 2 * n)
    mae = np.mean(abs(x), axis=1)
    return mae, lb, ub

def plot_score(reaction_index="1"):
    single_data = load_pickle("pickles/%s_single_result.pkl" % reaction_index)
    single_aux_data = load_pickle("pickles/%s_single_subset.pkl" % reaction_index)
    linear_data = load_pickle("pickles/%s_linear_result.pkl" % reaction_index)
    linear_positive_data = load_pickle("pickles/%s_linear_positive_result.pkl" % reaction_index)
    #markowitz_data = load_pickle("pickles/%s_markowitz_result.pkl" % reaction_index)
    markowitz_positive_data = load_pickle("pickles/%s_markowitz_positive_result.pkl" % reaction_index)
    print(single_data.keys())
    print(single_data['subset_names'])
    for i, errors in enumerate(linear_positive_data['errors']):
        print(single_data['subset_names'][i],np.mean(abs(errors)))

    quit()

    subset_names = single_data['subset_names']
    #TODO figure out how method names are generated
    #print(list(single_data.keys()))
    #quit()
    #print(subset_names)
    #for i in range(15):
    #    print(get_best_params(linear_data['cv_params'][i]))
    #quit()

    # plot gga
    #idx = isin(subset_names, 'hybrid', "+")
    idx = np.asarray([0,1,2,3,4,5]) + 6
    single_mae, single_lb, single_ub = get_mean_lower_and_upper_bound(abs(single_data['errors'][idx,:]), bootstrap=True, alpha=0.68)
    linear_mae = np.mean(abs(linear_data['errors'][idx,:]), axis=1)
    linear_positive_mae = np.mean(abs(linear_positive_data['errors'][idx,:]), axis=1)
    markowitz_positive_mae = np.mean(abs(markowitz_positive_data['errors'][idx,:]), axis=1)
    #markowitz_mae = np.exp(np.mean(np.log(abs(markowitz_data['errors'][idx,:])), axis=1))
    plt.fill_between(list(range(len(single_mae))), single_lb, single_ub, alpha=0.15)
    plt.plot(single_mae, "o-", label="single")
    plt.plot(linear_mae, "o-", label="linear")
    plt.plot(linear_positive_mae, "o-", label="linear_positive")
    #plt.plot(markowitz_mae, "o-", label="markowitz")
    plt.plot(markowitz_positive_mae, "o-", label="markowitz_positive")
    plt.legend()
    plt.show()

    #for idx, name in zip((idx1, idx2, idx3, idx4), 
    #        ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")):
    #    mae1 = np.mean(abs(data1['errors'][:,idx]), axis=1)
    #    mae2 = np.mean(abs(data2['errors'][:,idx]), axis=1)
    #    std1 = np.std(abs(data1['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))
    #    std2 = np.std(abs(data2['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))

    #    # Do the plot
    #    plt.fill_between(list(range(len(mae1))), mae1 - std1, mae1 + std1, alpha=0.15)
    #    plt.plot(mae1, "o-", label=label1)
    #    plt.fill_between(list(range(len(mae2))), mae2 - std2, mae2 + std2, alpha=0.15)
    #    plt.plot(mae2, "o-", label=label2)
    #    plt.ylabel("MAE (kcal/mol)\n")
    #    # Set 6 mae as upper range
    #    plt.ylim([0,6])
    #    plt.title(name)
    #    plt.legend()

    #    # Chemical accuracy line
    #    ax = plt.gca()
    #    #xmin, xmax = ax.get_xlim()
    #    #plt.plot([xmin, xmax], [1, 1], "--", c="k")
    #    ## In case the xlimit have changed, set it again
    #    #plt.xlim([xmin, xmax])

    #    # Set xtick labels
    #    ax.set_xticklabels(xlabels, rotation=-45, ha='left')

    #    if filename_base is not None:
    #        plt.savefig(filename_base + "_" + name.replace(" ", "_").lower() + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
    #        plt.clf()
    #    else:
    #        plt.show()


def plot_score2sns(file1, file2, label1=None, label2=None, filename_base=None):
    if label1 == None and label2 == None:
        label1 = file1.split("/")[-1].split(".")[0]
        label2 = file2.split("/")[-1].split(".")[0]
    data1 = load_pickle(file1)
    data2 = load_pickle(file2)


    xlabels = ["", "sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

    # rclass=1 (dissociation)
    idx1 = [3, 4, 5, 6, 7, 8, 17, 19, 27, 28, 29, 65, 72, 99, 100, 101]
    # rclass=2 (atom transfer)
    idx2 = [0, 11, 13, 15, 30, 32, 37, 40, 43, 45, 50, 52, 54, 67, 74, 76, 78, 84, 86, 88, 90, 92, 95, 97]
    # rclass=3 (dissociation barrier)
    idx3 = [10, 18, 20, 36, 39, 49, 66, 73]
    # rclass=4 (atom transfer barrier)
    idx4 = [1, 2, 9, 12, 14, 16, 21, 22, 23, 24, 25, 26, 31, 33, 34, 35, 38, 41, 42, 44, 46, 47, 48, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 75, 77, 79, 80, 81, 82, 83, 85, 87, 89, 91, 93, 94, 96, 98]

    for idx, name in zip((idx1, idx2, idx3, idx4), 
            ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")):
        mae1 = np.mean(abs(data1['errors'][:,idx]), axis=1)
        mae2 = np.mean(abs(data2['errors'][:,idx]), axis=1)

        # Create dataframe for the seaborn plots
        basis = []
        error = []
        method = []

        x1 = abs(data1['errors'][:,idx])
        x2 = abs(data2['errors'][:,idx])

        basis_list = ["sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                basis.append(basis_list[i])
                error.append(x1[i,j])
                method.append(label1)
                basis.append(basis_list[i])
                error.append(x2[i,j])
                method.append(label2)

        df = pd.DataFrame.from_dict({'basis': basis, 'MAE (kcal/mol)': error, 'method':method})
        sns.stripplot(x="basis", y="MAE (kcal/mol)", data=df, hue="method", jitter=0.1, dodge=True)


        plt.plot(mae1, "-", label=label1)
        plt.plot(mae2, "-", label=label2)
        # Set 6 mae as upper range
        plt.ylim([0,6])
        plt.xticks(rotation=-45, ha='left')
        plt.title(name)

        if filename_base is not None:
            plt.savefig(filename_base + "_" + name.replace(" ", "_").lower() + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
            plt.clf()
        else:
            plt.show()

def plot_score2(file1, file2, label1=None, label2=None, filename_base=None):
    if label1 == None and label2 == None:
        label1 = file1.split("/")[-1].split(".")[0]
        label2 = file2.split("/")[-1].split(".")[0]
    data1 = load_pickle(file1)
    data2 = load_pickle(file2)

    subset_names1 = data1['subset_names']

    # rclass=1 (dissociation)
    idx1 = [3, 4, 5, 6, 7, 8, 17, 19, 27, 28, 29, 65, 72, 99, 100, 101]
    # rclass=2 (atom transfer)
    idx2 = [0, 11, 13, 15, 30, 32, 37, 40, 43, 45, 50, 52, 54, 67, 74, 76, 78, 84, 86, 88, 90, 92, 95, 97]
    # rclass=3 (dissociation barrier)
    idx3 = [10, 18, 20, 36, 39, 49, 66, 73]
    # rclass=4 (atom transfer barrier)
    idx4 = [1, 2, 9, 12, 14, 16, 21, 22, 23, 24, 25, 26, 31, 33, 34, 35, 38, 41, 42, 44, 46, 47, 48, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 75, 77, 79, 80, 81, 82, 83, 85, 87, 89, 91, 93, 94, 96, 98]

    for idx, name in zip((idx1, idx2, idx3, idx4), 
            ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")):
        mae1 = np.mean(abs(data1['errors'][:,idx]), axis=1)
        mae2 = np.mean(abs(data2['errors'][:,idx]), axis=1)
        std1 = np.std(abs(data1['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))
        std2 = np.std(abs(data2['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))

        # Do the plot
        plt.fill_between(list(range(len(mae1))), mae1 - std1, mae1 + std1, alpha=0.15)
        plt.plot(mae1, "o-", label=label1)
        plt.fill_between(list(range(len(mae2))), mae2 - std2, mae2 + std2, alpha=0.15)
        plt.plot(mae2, "o-", label=label2)
        plt.ylabel("MAE (kcal/mol)\n")
        # Set 6 mae as upper range
        plt.ylim([0,6])
        plt.title(name)
        plt.legend()

        # Chemical accuracy line
        ax = plt.gca()
        #xmin, xmax = ax.get_xlim()
        #plt.plot([xmin, xmax], [1, 1], "--", c="k")
        ## In case the xlimit have changed, set it again
        #plt.xlim([xmin, xmax])

        # Set xtick labels
        ax.set_xticklabels(xlabels, rotation=-45, ha='left')

        if filename_base is not None:
            plt.savefig(filename_base + "_" + name.replace(" ", "_").lower() + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
            plt.clf()
        else:
            plt.show()

def plot_score3sns(file1, file2, file3, label1=None, label2=None, label3=None, filename_base=None):
    if label1 == None and label2 == None and label3 == None:
        label1 = file1.split("/")[-1].split(".")[0]
        label2 = file2.split("/")[-1].split(".")[0]
        label3 = file3.split("/")[-1].split(".")[0]
    data1 = load_pickle(file1)
    data2 = load_pickle(file2)
    data3 = load_pickle(file3)

    xlabels = ["", "sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

    # rclass=1 (dissociation)
    idx1 = [3, 4, 5, 6, 7, 8, 17, 19, 27, 28, 29, 65, 72, 99, 100, 101]
    # rclass=2 (atom transfer)
    idx2 = [0, 11, 13, 15, 30, 32, 37, 40, 43, 45, 50, 52, 54, 67, 74, 76, 78, 84, 86, 88, 90, 92, 95, 97]
    # rclass=3 (dissociation barrier)
    idx3 = [10, 18, 20, 36, 39, 49, 66, 73]
    # rclass=4 (atom transfer barrier)
    idx4 = [1, 2, 9, 12, 14, 16, 21, 22, 23, 24, 25, 26, 31, 33, 34, 35, 38, 41, 42, 44, 46, 47, 48, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 75, 77, 79, 80, 81, 82, 83, 85, 87, 89, 91, 93, 94, 96, 98]

    for idx, name in zip((idx1, idx2, idx3, idx4), 
            ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")):
        mae1 = np.mean(abs(data1['errors'][:,idx]), axis=1)
        mae2 = np.mean(abs(data2['errors'][:,idx]), axis=1)
        mae3 = np.mean(abs(data3['errors'][:,idx]), axis=1)

        # Create dataframe for the seaborn plots
        basis = []
        error = []
        method = []

        x1 = abs(data1['errors'][:,idx])
        x2 = abs(data2['errors'][:,idx])
        x3 = abs(data3['errors'][:,idx])

        basis_list = ["sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                basis.append(basis_list[i])
                error.append(x1[i,j])
                method.append(label1)
                basis.append(basis_list[i])
                error.append(x2[i,j])
                method.append(label2)
                basis.append(basis_list[i])
                error.append(x3[i,j])
                method.append(label3)

        df = pd.DataFrame.from_dict({'basis': basis, 'MAE (kcal/mol)': error, 'method':method})
        sns.stripplot(x="basis", y="MAE (kcal/mol)", data=df, hue="method", jitter=0.1, dodge=True)

        plt.plot(mae1, "-", label=label1)
        plt.plot(mae2, "-", label=label2)
        plt.plot(mae3, "-", label=label3)
        # Set 6 mae as upper range
        plt.ylim([0,6])
        plt.xticks(rotation=-45, ha='left')
        plt.title(name)

        if filename_base is not None:
            plt.savefig(filename_base + "_" + name.replace(" ", "_").lower() + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
            plt.clf()
        else:
            plt.show()

def plot_score3(file1, file2, file3, label1=None, label2=None, label3=None, filename_base=None):
    if label1 == None and label2 == None and label3 == None:
        label1 = file1.split("/")[-1].split(".")[0]
        label2 = file2.split("/")[-1].split(".")[0]
        label3 = file3.split("/")[-1].split(".")[0]
    data1 = load_pickle(file1)
    data2 = load_pickle(file2)
    data3 = load_pickle(file3)

    xlabels = ["", "sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

    # rclass=1 (dissociation)
    idx1 = [3, 4, 5, 6, 7, 8, 17, 19, 27, 28, 29, 65, 72, 99, 100, 101]
    # rclass=2 (atom transfer)
    idx2 = [0, 11, 13, 15, 30, 32, 37, 40, 43, 45, 50, 52, 54, 67, 74, 76, 78, 84, 86, 88, 90, 92, 95, 97]
    # rclass=3 (dissociation barrier)
    idx3 = [10, 18, 20, 36, 39, 49, 66, 73]
    # rclass=4 (atom transfer barrier)
    idx4 = [1, 2, 9, 12, 14, 16, 21, 22, 23, 24, 25, 26, 31, 33, 34, 35, 38, 41, 42, 44, 46, 47, 48, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 75, 77, 79, 80, 81, 82, 83, 85, 87, 89, 91, 93, 94, 96, 98]

    for idx, name in zip((idx1, idx2, idx3, idx4), 
            ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")):
        mae1 = np.mean(abs(data1['errors'][:,idx]), axis=1)
        mae2 = np.mean(abs(data2['errors'][:,idx]), axis=1)
        mae3 = np.mean(abs(data3['errors'][:,idx]), axis=1)
        std1 = np.std(abs(data1['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))
        std2 = np.std(abs(data2['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))
        std3 = np.std(abs(data3['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))

        # Do the plot
        plt.fill_between(list(range(len(mae1))), mae1 - std1, mae1 + std1, alpha=0.15)
        plt.plot(mae1, "o-", label=label1)
        plt.fill_between(list(range(len(mae2))), mae2 - std2, mae2 + std2, alpha=0.15)
        plt.plot(mae2, "o-", label=label2)
        plt.fill_between(list(range(len(mae3))), mae3 - std3, mae3 + std3, alpha=0.15)
        plt.plot(mae3, "o-", label=label3)
        plt.ylabel("MAE (kcal/mol)\n")
        # Set 6 mae as upper range
        plt.ylim([0,6])
        plt.title(name)
        plt.legend()

        # Chemical accuracy line
        ax = plt.gca()
        #xmin, xmax = ax.get_xlim()
        #plt.plot([xmin, xmax], [1, 1], "--", c="k")
        ## In case the xlimit have changed, set it again
        #plt.xlim([xmin, xmax])

        # Set xtick labels
        ax.set_xticklabels(xlabels, rotation=-45, ha='left')

        if filename_base is not None:
            plt.savefig(filename_base + "_" + name.replace(" ", "_").lower() + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
            plt.clf()
        else:
            plt.show()

def plot_score4sns(file1, file2, file3, file4, label1=None, label2=None, label3=None, label4=None, filename_base=None):
    if label1 == None and label2 == None and label3 == None and label4 == None:
        label1 = file1.split("/")[-1].split(".")[0]
        label2 = file2.split("/")[-1].split(".")[0]
        label3 = file3.split("/")[-1].split(".")[0]
        label3 = file3.split("/")[-1].split(".")[0]
    data1 = load_pickle(file1)
    data2 = load_pickle(file2)
    data3 = load_pickle(file3)
    data4 = load_pickle(file4)

    xlabels = ["", "sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

    # rclass=1 (dissociation)
    idx1 = [3, 4, 5, 6, 7, 8, 17, 19, 27, 28, 29, 65, 72, 99, 100, 101]
    # rclass=2 (atom transfer)
    idx2 = [0, 11, 13, 15, 30, 32, 37, 40, 43, 45, 50, 52, 54, 67, 74, 76, 78, 84, 86, 88, 90, 92, 95, 97]
    # rclass=3 (dissociation barrier)
    idx3 = [10, 18, 20, 36, 39, 49, 66, 73]
    # rclass=4 (atom transfer barrier)
    idx4 = [1, 2, 9, 12, 14, 16, 21, 22, 23, 24, 25, 26, 31, 33, 34, 35, 38, 41, 42, 44, 46, 47, 48, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 75, 77, 79, 80, 81, 82, 83, 85, 87, 89, 91, 93, 94, 96, 98]

    for idx, name in zip((idx1, idx2, idx3, idx4), 
            ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")):
        mae1 = np.mean(abs(data1['errors'][:,idx]), axis=1)
        mae2 = np.mean(abs(data2['errors'][:,idx]), axis=1)
        mae3 = np.mean(abs(data3['errors'][:,idx]), axis=1)
        mae4 = np.mean(abs(data4['errors'][:,idx]), axis=1)

        # Create dataframe for the seaborn plots
        basis = []
        error = []
        method = []

        x1 = abs(data1['errors'][:,idx])
        x2 = abs(data2['errors'][:,idx])
        x3 = abs(data3['errors'][:,idx])
        x4 = abs(data4['errors'][:,idx])

        basis_list = ["sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                basis.append(basis_list[i])
                error.append(x1[i,j])
                method.append(label1)
                basis.append(basis_list[i])
                error.append(x2[i,j])
                method.append(label2)
                basis.append(basis_list[i])
                error.append(x3[i,j])
                method.append(label3)
                basis.append(basis_list[i])
                error.append(x4[i,j])
                method.append(label4)

        df = pd.DataFrame.from_dict({'basis': basis, 'MAE (kcal/mol)': error, 'method':method})
        sns.stripplot(x="basis", y="MAE (kcal/mol)", data=df, hue="method", jitter=0.1, dodge=True)

        plt.plot(mae1, "-", label=label1)
        plt.plot(mae2, "-", label=label2)
        plt.plot(mae3, "-", label=label3)
        plt.plot(mae4, "-", label=label4)
        # Set 6 mae as upper range
        plt.ylim([0,6])
        plt.xticks(rotation=-45, ha='left')
        plt.title(name)

        if filename_base is not None:
            plt.savefig(filename_base + "_" + name.replace(" ", "_").lower() + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
            plt.clf()
        else:
            plt.show()

def plot_score4(file1, file2, file3, file4, label1=None, label2=None, label3=None, label4=None, filename_base=None):
    if label1 == None and label2 == None and label3 == None and label4 == None:
        label1 = file1.split("/")[-1].split(".")[0]
        label2 = file2.split("/")[-1].split(".")[0]
        label3 = file3.split("/")[-1].split(".")[0]
        label4 = file4.split("/")[-1].split(".")[0]
    data1 = load_pickle(file1)
    data2 = load_pickle(file2)
    data3 = load_pickle(file3)
    data4 = load_pickle(file4)

    xlabels = ["", "sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

    # rclass=1 (dissociation)
    idx1 = [3, 4, 5, 6, 7, 8, 17, 19, 27, 28, 29, 65, 72, 99, 100, 101]
    # rclass=2 (atom transfer)
    idx2 = [0, 11, 13, 15, 30, 32, 37, 40, 43, 45, 50, 52, 54, 67, 74, 76, 78, 84, 86, 88, 90, 92, 95, 97]
    # rclass=3 (dissociation barrier)
    idx3 = [10, 18, 20, 36, 39, 49, 66, 73]
    # rclass=4 (atom transfer barrier)
    idx4 = [1, 2, 9, 12, 14, 16, 21, 22, 23, 24, 25, 26, 31, 33, 34, 35, 38, 41, 42, 44, 46, 47, 48, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 75, 77, 79, 80, 81, 82, 83, 85, 87, 89, 91, 93, 94, 96, 98]

    for idx, name in zip((idx1, idx2, idx3, idx4), 
            ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")):
        mae1 = np.mean(abs(data1['errors'][:,idx]), axis=1)
        mae2 = np.mean(abs(data2['errors'][:,idx]), axis=1)
        mae3 = np.mean(abs(data3['errors'][:,idx]), axis=1)
        mae4 = np.mean(abs(data4['errors'][:,idx]), axis=1)
        std1 = np.std(abs(data1['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))
        std2 = np.std(abs(data2['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))
        std3 = np.std(abs(data3['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))
        std4 = np.std(abs(data4['errors'][:,idx]), axis=1, ddof=1)/np.sqrt(len(idx))

        # Do the plot
        #plt.fill_between(list(range(len(mae1))), mae1 - std1, mae1 + std1, alpha=0.15)
        plt.plot(mae1, "o-", label=label1)
        #plt.fill_between(list(range(len(mae2))), mae2 - std2, mae2 + std2, alpha=0.15)
        plt.plot(mae2, "o-", label=label2)
        #plt.fill_between(list(range(len(mae3))), mae3 - std3, mae3 + std3, alpha=0.15)
        plt.plot(mae3, "o-", label=label3)
        #plt.fill_between(list(range(len(mae4))), mae4 - std4, mae4 + std4, alpha=0.15)
        plt.plot(mae4, "o-", label=label4)
        plt.ylabel("MAE (kcal/mol)\n")
        # Set 6 mae as upper range
        plt.ylim([0,6])
        plt.title(name)
        plt.legend()

        # Chemical accuracy line
        ax = plt.gca()
        #xmin, xmax = ax.get_xlim()
        #plt.plot([xmin, xmax], [1, 1], "--", c="k")
        ## In case the xlimit have changed, set it again
        #plt.xlim([xmin, xmax])

        # Set xtick labels
        ax.set_xticklabels(xlabels, rotation=-45, ha='left')

        if filename_base is not None:
            plt.savefig(filename_base + "_" + name.replace(" ", "_").lower() + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
            plt.clf()
        else:
            plt.show()

def plot_distribution3(file1, file2, file3, label1=None, label2=None, label3=None, 
        idx=0, method=0, filename_base=None):

    def kde(x, xgrid):
        return gaussian_kde(x).evaluate(xgrid)


    if label1 == None and label2 == None and label3 == None:
        label1 = file1.split("/")[-1].split(".")[0]
        label2 = file2.split("/")[-1].split(".")[0]
        label3 = file3.split("/")[-1].split(".")[0]
    data1 = load_pickle(file1)
    data2 = load_pickle(file2)
    data3 = load_pickle(file3)

    xlabels = ["", "sto-3g", "sv(p)", "svp/6-31+G(d,p)", "avdz", "tzvp", "avtz", "qzvp", "WF"]

    # rclass=1 (dissociation)
    idx1 = [3, 4, 5, 6, 7, 8, 17, 19, 27, 28, 29, 65, 72, 99, 100, 101]
    # rclass=2 (atom transfer)
    idx2 = [0, 11, 13, 15, 30, 32, 37, 40, 43, 45, 50, 52, 54, 67, 74, 76, 78, 84, 86, 88, 90, 92, 95, 97]
    # rclass=3 (dissociation barrier)
    idx3 = [10, 18, 20, 36, 39, 49, 66, 73]
    # rclass=4 (atom transfer barrier)
    idx4 = [1, 2, 9, 12, 14, 16, 21, 22, 23, 24, 25, 26, 31, 33, 34, 35, 38, 41, 42, 44, 46, 47, 48, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 75, 77, 79, 80, 81, 82, 83, 85, 87, 89, 91, 93, 94, 96, 98]

    idx_names = ("Dissociation", "Atom transfer", "Dissociation barrier", "Atom transfer barrier")

    indices = np.asarray([idx1, idx2, idx3, idx4][idx])

    title = idx_names[idx] + " / " + xlabels[method+1]

    x1 = data1['errors'][method,indices]
    x2 = data2['errors'][method,indices]
    x3 = data3['errors'][method,indices]

    w1 = np.where(abs(x1) > 5)[0]
    w2 = np.where(abs(x2) > 5)[0]
    w3 = np.where(abs(x3) > 5)[0]
    print(data1['reaction_names'][indices[w1]])
    print(data1['reaction_names'][indices[w2]])
    print(data1['reaction_names'][indices[w3]])

    xgrid = np.linspace(min(min(x1),min(x2),min(x3))-3, max(max(x1),max(x2),max(x3))+3, 1000)

    y1 = kde(x1, xgrid)
    y2 = kde(x2, xgrid)
    y3 = kde(x3, xgrid)

    plt.fill_between(xgrid, np.zeros(xgrid.size), y1, alpha=0.1)
    plt.plot(xgrid, y1, label=label1)
    plt.fill_between(xgrid, np.zeros(xgrid.size), y2, alpha=0.1)
    plt.plot(xgrid, y2, label=label2)
    plt.fill_between(xgrid, np.zeros(xgrid.size), y3, alpha=0.1)
    plt.plot(xgrid, y3, label=label3)
    plt.xlim([xgrid[0], xgrid[-1]])
    plt.legend()
    plt.title(title)

    if filename_base is not None:
        plt.savefig(filename_base + ".pdf", pad_inches=0.0, bbox_inches = "tight", dpi = 300)
        plt.clf()
    else:
        plt.show()

if __name__ == "__main__":

    plot_score("7")

    #plot_score2("pickles/single_method_all_reactions_result.pkl", "pickles/single_method_same_reactions_result.pkl",
    #        label1="single_all", label2="single_same", filename_base='single_all_vs_same')

    #plot_score4("pickles/linear_method_all_reactions_result.pkl", "pickles/linear_method_positive_all_reactions_result.pkl",
    #        "pickles/linear_method_positive_same_reactions_result.pkl", "pickles/linear_method_same_reactions_result.pkl",
    #        label1="linear_all", label2="linear_all_pos", label3="linear_same_pos", label4="linear_same", filename_base='linear')

    #plot_score4("pickles/markowitz_all_reactions_result.pkl", "pickles/markowitz_positive_all_reactions_result.pkl",
    #        "pickles/markowitz_positive_same_reactions_result.pkl", "pickles/markowitz_same_reactions_result.pkl",
    #        label1="markowitz_all", label2="markowits_all_pos", label3="markowitz_same_pos", label4="markowitz_same", filename_base='markowitz')

    #plot_score3("pickles/single_method_same_reactions_result.pkl", "pickles/linear_method_same_reactions_result.pkl",
    #        "pickles/markowitz_same_reactions_result.pkl",
    #        label1="single", label2="linear", label3="markowitz", filename_base='comparison')


    #plot_score2("pickles/less_strict_single_method_all_reactions_result.pkl", "pickles/less_strict_single_method_same_reactions_result.pkl",
    #        label1="single_all", label2="single_same", filename_base='less_strict_single_all_vs_same')

    #plot_score4("pickles/less_strict_linear_method_all_reactions_result.pkl", "pickles/less_strict_linear_method_positive_all_reactions_result.pkl",
    #        "pickles/less_strict_linear_method_positive_same_reactions_result.pkl", "pickles/less_strict_linear_method_same_reactions_result.pkl",
    #        label1="linear_all", label2="linear_all_pos", label3="linear_same_pos", label4="linear_same", filename_base='less_strict_linear')

    #plot_score4("pickles/less_strict_markowitz_all_reactions_result.pkl", "pickles/less_strict_markowitz_positive_all_reactions_result.pkl",
    #        "pickles/less_strict_markowitz_positive_same_reactions_result.pkl", "pickles/less_strict_markowitz_same_reactions_result.pkl",
    #        label1="markowitz_all", label2="markowitz_all_pos", label3="markowitz_same_pos", label4="markowitz_same", filename_base='less_strict_markowitz')

    #plot_score3("pickles/less_strict_single_method_same_reactions_result.pkl", "pickles/less_strict_linear_method_same_reactions_result.pkl",
    #        "pickles/less_strict_markowitz_same_reactions_result.pkl",
    #        label1="single", label2="linear", label3="markowitz", filename_base='less_strict_comparison')

    #plot_distribution3("pickles/single_method_same_reactions_result.pkl",
    #        "pickles/linear_method_same_reactions_result.pkl", "pickles/markowitz_same_reactions_result.pkl",
    #        label1="single", label2="linear", label3="markowitz", idx=3, method=2, filename_base='distribution1')

    #plot_distribution3("pickles/single_method_same_reactions_result.pkl",
    #        "pickles/linear_method_same_reactions_result.pkl", "pickles/markowitz_same_reactions_result.pkl",
    #        label1="single", label2="linear", label3="markowitz", idx=3, method=3, filename_base='distribution2')

    #plot_score2sns("pickles/single_method_all_reactions_result.pkl", "pickles/single_method_same_reactions_result.pkl",
    #        label1="single_all", label2="single_same", filename_base='sns_single_all_vs_same')

    #plot_score4sns("pickles/linear_method_all_reactions_result.pkl", "pickles/linear_method_positive_all_reactions_result.pkl",
    #        "pickles/linear_method_positive_same_reactions_result.pkl", "pickles/linear_method_same_reactions_result.pkl",
    #        label1="linear_all", label2="linear_all_pos", label3="linear_same_pos", label4="linear_same", filename_base='sns_linear')

    #plot_score4sns("pickles/markowitz_all_reactions_result.pkl", "pickles/markowitz_positive_all_reactions_result.pkl",
    #        "pickles/markowitz_positive_same_reactions_result.pkl", "pickles/markowitz_same_reactions_result.pkl",
    #        label1="markowitz_all", label2="markowits_all_pos", label3="markowitz_same_pos", label4="markowitz_same", filename_base='sns_markowitz')

    #plot_score3sns("pickles/single_method_same_reactions_result.pkl", "pickles/linear_method_same_reactions_result.pkl",
    #        "pickles/markowitz_same_reactions_result.pkl",
    #        label1="single", label2="linear", label3="markowitz", filename_base='sns_comparison')


    #plot_score2sns("pickles/less_strict_single_method_all_reactions_result.pkl", "pickles/less_strict_single_method_same_reactions_result.pkl",
    #        label1="single_all", label2="single_same", filename_base='sns_less_strict_single_all_vs_same')

    #plot_score4sns("pickles/less_strict_linear_method_all_reactions_result.pkl", "pickles/less_strict_linear_method_positive_all_reactions_result.pkl",
    #        "pickles/less_strict_linear_method_positive_same_reactions_result.pkl", "pickles/less_strict_linear_method_same_reactions_result.pkl",
    #        label1="linear_all", label2="linear_all_pos", label3="linear_same_pos", label4="linear_same", filename_base='sns_less_strict_linear')

    #plot_score4sns("pickles/less_strict_markowitz_all_reactions_result.pkl", "pickles/less_strict_markowitz_positive_all_reactions_result.pkl",
    #        "pickles/less_strict_markowitz_positive_same_reactions_result.pkl", "pickles/less_strict_markowitz_same_reactions_result.pkl",
    #        label1="markowitz_all", label2="markowitz_all_pos", label3="markowitz_same_pos", label4="markowitz_same", filename_base='sns_less_strict_markowitz')

    #plot_score3sns("pickles/less_strict_single_method_same_reactions_result.pkl", "pickles/less_strict_linear_method_same_reactions_result.pkl",
    #        "pickles/less_strict_markowitz_same_reactions_result.pkl",
    #        label1="single", label2="linear", label3="markowitz", filename_base='sns_less_strict_comparison')

