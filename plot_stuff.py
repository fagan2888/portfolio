import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rc
import seaborn as sns
import os

# Add the module to source path
dirname = os.path.dirname(os.path.realpath(__file__))
source_path = dirname + "/portfolio"
sys.path.append(source_path)

from portfolio.plotting import plot_comparison


def load_pickle(filename):
    with open(filename, "rb") as f:
        d = pickle.load(f)
    return d

# set matplotlib defaults
sns.set(font_scale=1.0)
sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})
rc('text', usetex=False)

def plot_score2(file1, file2, label1=None, label2=None, filename_base=None):
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


if __name__ == "__main__":
    plot_score2("pickles/single_method_all_reactions_result.pkl", "pickles/single_method_same_reactions_result.pkl",
            label1="single_all", label2="single_same", filename_base='single_all_vs_same')

    plot_score4("pickles/linear_method_all_reactions_result.pkl", "pickles/linear_method_positive_all_reactions_result.pkl",
            "pickles/linear_method_positive_same_reactions_result.pkl", "pickles/linear_method_same_reactions_result.pkl",
            label1="linear_all", label2="linear_all_pos", label3="linear_same_pos", label4="linear_same", filename_base='linear')

    plot_score4("pickles/markowitz_all_reactions_result.pkl", "pickles/markowitz_positive_all_reactions_result.pkl",
            "pickles/markowitz_positive_same_reactions_result.pkl", "pickles/markowitz_same_reactions_result.pkl",
            label1="markowitz_all", label2="markowits_all_pos", label3="markowitz_same_pos", label4="markowitz_same", filename_base='markowitz')

    plot_score3("pickles/single_method_same_reactions_result.pkl", "pickles/linear_method_same_reactions_result.pkl",
            "pickles/markowitz_same_reactions_result.pkl",
            label1="single", label2="linear", label3="markowitz", filename_base='comparison')


    plot_score2("pickles/less_strict_single_method_all_reactions_result.pkl", "pickles/less_strict_single_method_same_reactions_result.pkl",
            label1="single_all", label2="single_same", filename_base='less_strict_single_all_vs_same')

    plot_score4("pickles/less_strict_linear_method_all_reactions_result.pkl", "pickles/less_strict_linear_method_positive_all_reactions_result.pkl",
            "pickles/less_strict_linear_method_positive_same_reactions_result.pkl", "pickles/less_strict_linear_method_same_reactions_result.pkl",
            label1="linear_all", label2="linear_all_pos", label3="linear_same_pos", label4="linear_same", filename_base='less_strict_linear')

    plot_score4("pickles/less_strict_markowitz_all_reactions_result.pkl", "pickles/less_strict_markowitz_positive_all_reactions_result.pkl",
            "pickles/less_strict_markowitz_positive_same_reactions_result.pkl", "pickles/less_strict_markowitz_same_reactions_result.pkl",
            label1="markowitz_all", label2="markowitz_all_pos", label3="markowitz_same_pos", label4="markowitz_same", filename_base='less_strict_markowitz')

    plot_score3("pickles/less_strict_single_method_same_reactions_result.pkl", "pickles/less_strict_linear_method_same_reactions_result.pkl",
            "pickles/less_strict_markowitz_same_reactions_result.pkl",
            label1="single", label2="linear", label3="markowitz", filename_base='less_strict_comparison')
