import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from scipy.interpolate import make_interp_spline
import seaborn as sns; sns.set_theme(color_codes=True)
import os
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.metrics
import sklearn.datasets
import random
import warnings
import argparse
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set_style(style='white')

all_labels = dict()

#Shuffle TADs With Random Labels 
def shuffle_tad_labels(tads):
    rand_labels = tads.copy()
    labels = []
    for ch in all_labels:
        tad_ch = rand_labels[rand_labels["chr1"] == ch]
        rand_categories = all_labels[ch].copy()
        random.shuffle(rand_categories)
        random_labels = []
        for j in range(len(rand_categories)):
            cat = rand_categories[j]
            lst = cat.split("-")
            if j == len(rand_categories)-1:
                random_labels.append(lst[0])
                random_labels.append(lst[1])
            else:
                random_labels.append(lst[0])
        ind = 0
        for i, row in tad_ch.iterrows():
            if "Boundary" in row["Label"]:
                labels.append("Boundary")
            else:
                labels.append(random_labels[ind])
                ind += 1
    rand_labels["Label"] = labels
    return rand_labels
        

def count_clumped_meqtls(chrom, x1, x2, clumped_meqtls):
    common_chr = clumped_meqtls[clumped_meqtls["CHR"] == str(chrom)]
    snps = common_chr[(common_chr["BP"] >= x1) & (common_chr["BP"] <= x2)]
    return len(snps)

# Bin the Genomic Space Between Two TADs in 40 Equal Bins and Calculate Normalized Burden of meQTLs in Each Bin
def consecutive_tad_distribution(tads, clumped_meqtls):
    dic = dict()
    active_active_normalized = []
    active_inactive_normalized = []
    inactive_active_normalized = []
    inactive_inactive_normalized = []
    num_bins = 40
    start_boundary = []
    end_boundary = []

    for i in range(num_bins):
        active_active_normalized.append([])
        active_inactive_normalized.append([])
        inactive_active_normalized.append([])
        inactive_inactive_normalized.append([])

    for ind in range(0, len(tads)):
#         print(ind)
#         if ind%105 == 0:
#             print(ind)
        if ind+3>=len(tads):
            break
        if tads.iloc[ind]["chr1"] != tads.iloc[ind+3]["chr1"]:
            continue
        if "Boundary" in tads.iloc[ind]["Label"]:
            continue
        bin_size = (tads.iloc[ind+3]["x2"]-tads.iloc[ind]["x1"])//(num_bins)
        #bin_size = 25000
        temp = tads.iloc[ind]["Label"]
        label1 = tads.iloc[ind]["Label"]
        label2 = tads.iloc[ind+3]["Label"]
        if label1 == "Mixed" or label2 == "Mixed":
            continue
        start = tads.iloc[ind]["x1"]
        inx = 0
        while (start<=tads.iloc[ind+3]["x2"]-bin_size):
            if start>=tads.iloc[ind]["x2"] and start<tads.iloc[ind+2]["x1"]:
                if "Boundary" not in temp:
                    start_boundary.append(inx)
                temp = "Boundary"
            elif start+bin_size<=tads.iloc[ind]["x2"]:
                temp = tads.iloc[ind]["Label"]
            elif start >= tads.iloc[ind+3]["x1"]:
                if temp == "Boundary":
                    end_boundary.append(inx)
                temp = tads.iloc[ind+3]["Label"]
            if label1 == "Active" and label2 == "Active":
                if inx<len(active_active_normalized):
                    active_active_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
                else:
                    active_active_normalized.append([])
                    active_active_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
            if label1 == "Active" and label2 == "Inactive":
                if inx<len(active_inactive_normalized):
                    active_inactive_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
                else:
                    active_inactive_normalized.append([])
                    active_inactive_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
            if label1 == "Inactive" and label2 == "Active":
                if inx<len(inactive_active_normalized):
                    inactive_active_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
                else:
                    inactive_active_normalized.append([])
                    inactive_active_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
            if label1 == "Inactive" and label2 == "Inactive":
                if inx<len(inactive_inactive_normalized):
                    inactive_inactive_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
                else:
                    inactive_inactive_normalized.append([])
                    inactive_inactive_normalized[inx].append(count_clumped_meqtls(tads.iloc[ind]["chr1"], start, start+bin_size, clumped_meqtls)*(1000)/(bin_size))
            start = start + bin_size
            inx+=1
    #     break
    return active_active_normalized, active_inactive_normalized, inactive_active_normalized, inactive_inactive_normalized, start_boundary, end_boundary



def main(args):
    common_TADs_meQTLs = pd.read_csv("/cellar/users/sgoudarzi/common_TADs_proper_annot.csv")
    new_x1 = []
    new_x2 = []
    for i, row in common_TADs_meQTLs.iterrows():
        new_x1.append(int(row["x1"])/1000)
        new_x2.append(int(row["x2"])/1000)
    common_TADs_meQTLs["x1"] = new_x1
    common_TADs_meQTLs["x2"] = new_x2
    common_TADs_meQTLs = common_TADs_meQTLs.sort_values(["chr1", "x1"])
    common_TADs_meQTLs = common_TADs_meQTLs.reset_index(drop=True)
    common_TADs_meQTLs_noboundary = common_TADs_meQTLs[~common_TADs_meQTLs["Label"].str.contains("Boundary")]
    common_TADs_meQTLs_noboundary = common_TADs_meQTLs_noboundary.reset_index(drop=True)
    for ch in list(pd.unique(common_TADs_meQTLs_noboundary["chr1"])):
        all_labels[ch] = []
    for i, row in common_TADs_meQTLs_noboundary.iterrows():
        if i<len(common_TADs_meQTLs_noboundary)-1:
            if row["chr1"] == common_TADs_meQTLs_noboundary.iloc[i+1]["chr1"]:
                all_labels[row["chr1"]].append(row["Label"]+"-"+common_TADs_meQTLs_noboundary.iloc[i+1]["Label"])
    
    tad_lengths = []
    for i, row in common_TADs_meQTLs.iterrows():
        tad_lengths.append(row["x2"]-row["x1"])
    common_TADs_meQTLs["Length"] = tad_lengths
    clumped_meqtls = pd.read_csv("/cellar/users/sgoudarzi/all_meqtls.clumped", delim_whitespace=True)
    clumped_meqtls["CHR"] = clumped_meqtls["CHR"].astype(str)
    
    # Run the Distribution Analysis on Randomly Shuffle TAD Labels in 100 Trials
    active_active_normalized_total = []
    active_inactive_normalized_total = []
    inactive_active_normalized_total = []
    inactive_inactive_normalized_total = []
    start_boundary = 0
    end_boundary = 0
    num_trials = 100
    active_active_diff_shuffled = []
    active_inactive_diff_shuffled = []
    inactive_active_diff_shuffled = []
    inactive_inactive_diff_shuffled = []
    for trial in range(num_trials):
        #print(trial)
        active_active_normalized = []
        active_inactive_normalized = []
        inactive_active_normalized = []
        inactive_inactive_normalized = []
        random_tads = shuffle_tad_labels(common_TADs_meQTLs)
        active_active_normalized, active_inactive_normalized, inactive_active_normalized, inactive_inactive_normalized, start_boundary, end_boundary = consecutive_tad_distribution(random_tads, clumped_meqtls)
        
        active_active_diff_shuffled.append(np.mean(active_active_normalized[int(np.mean(end_boundary))]) - np.mean(active_active_normalized[int(np.mean(start_boundary))]))
        active_inactive_diff_shuffled.append(np.mean(active_inactive_normalized[int(np.mean(end_boundary))]) - np.mean(active_inactive_normalized[int(np.mean(start_boundary))]))
        inactive_active_diff_shuffled.append(np.mean(inactive_active_normalized[int(np.mean(end_boundary))]) - np.mean(inactive_active_normalized[int(np.mean(start_boundary))]))
        inactive_inactive_diff_shuffled.append(np.mean(inactive_inactive_normalized[int(np.mean(end_boundary))]) - np.mean(inactive_inactive_normalized[int(np.mean(start_boundary))]))
        
        active_active_normalized_total.append(active_active_normalized)
        active_inactive_normalized_total.append(active_inactive_normalized)
        inactive_active_normalized_total.append(inactive_active_normalized)
        inactive_inactive_normalized_total.append(inactive_inactive_normalized)
    
    # Run the Distribution Analysis on the True TAD Label
    active_active_normalized_true = []
    active_inactive_normalized_true = []
    inactive_active_normalized_true = []
    inactive_inactive_normalized_true = []
    active_active_normalized_true, active_inactive_normalized_true, inactive_active_normalized_true, inactive_inactive_normalized_true, start_boundary, end_boundary = consecutive_tad_distribution(common_TADs_meQTLs, clumped_meqtls)
    
    active_active_diff = np.mean(active_active_normalized_true[int(np.mean(end_boundary))]) - np.mean(active_active_normalized_true[int(np.mean(start_boundary))])
    active_inactive_diff = np.mean(active_inactive_normalized_true[int(np.mean(end_boundary))]) - np.mean(active_inactive_normalized_true[int(np.mean(start_boundary))])
    inactive_active_diff = np.mean(inactive_active_normalized_true[int(np.mean(end_boundary))]) - np.mean(inactive_active_normalized_true[int(np.mean(start_boundary))])
    inactive_inactive_diff = np.mean(inactive_inactive_normalized_true[int(np.mean(end_boundary))]) - np.mean(inactive_inactive_normalized_true[int(np.mean(start_boundary))])
    
    #Compare the Difference in Burden of meQTLs Between Two Boundaries of the Actual and the Randomized TADs
    #Use a 1-sample Student-T test with the Population Mean the True Mean in Difference of Burden Between TADs
    
    print("Active-Active T-Test: ", stats.ttest_1samp(active_active_diff_shuffled, popmean=active_active_diff))
    print("Active-Inactive T-Test: ", stats.ttest_1samp(active_inactive_diff_shuffled, popmean=active_inactive_diff))
    print("Inactive-Active T-Test: ", stats.ttest_1samp(inactive_active_diff_shuffled, popmean=inactive_active_diff))
    print("Inactive-Inactive T-Test: ", stats.ttest_1samp(inactive_inactive_diff_shuffled, popmean=inactive_inactive_diff))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)