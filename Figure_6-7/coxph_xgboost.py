import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import MinMaxScaler
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import argparse
from lifelines import CoxPHFitter
import statsmodels.api as sm
import statsmodels.stats.multitest as multi
from scipy.stats import mannwhitneyu
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def make_df(tumor, surv,surv_type,raw, sig_snps):
    #get survival dataframe
    surv=surv.rename(columns={"bcr_patient_barcode":"FID"})
    #get genotypes
    snps=pd.read_csv(raw,delimiter=" ")
    surv_snps = pd.unique(sig_snps["snps"])
    for col in snps:
        if "_" in col:
            tmp = col.split("_")[0]
            if tmp not in surv_snps:
                del snps[col]
    cols=[x for x in snps.columns if x not in ["IID","PAT","MAT","SEX","PHENOTYPE"]]
    snps=snps[cols]
    has_tumor = False
    temp_surv=pd.merge(surv[["FID",surv_type,surv_type+".time","age_at_initial_pathologic_diagnosis","gender","ajcc_pathologic_tumor_stage"]], snps,on="FID")
    temp_surv=temp_surv[temp_surv["ajcc_pathologic_tumor_stage"].str.contains("Stage")]
    temp_surv=temp_surv[~(temp_surv["ajcc_pathologic_tumor_stage"].isin(["Stage X"]))]
    if (len(temp_surv)>=100):
        has_tumor = True
    full_surv = pd.DataFrame()
    if (has_tumor):
        #combine dataframes
        if tumor == "BRCA":
            full_surv=pd.merge(surv[["FID",surv_type,surv_type+".time","age_at_initial_pathologic_diagnosis","gender","ajcc_pathologic_tumor_stage", "breast_carcinoma_estrogen_receptor_status", "breast_carcinoma_progesterone_receptor_status", "lab_proc_her2_neu_immunohistochemistry_receptor_status"]], snps,on="FID")
        elif tumor == "HNSC":
            full_surv=pd.merge(surv[["FID",surv_type,surv_type+".time","age_at_initial_pathologic_diagnosis","gender","ajcc_pathologic_tumor_stage", "hpv_status_by_p16_testing"]], snps,on="FID")
        else:
            full_surv=pd.merge(surv[["FID",surv_type,surv_type+".time","age_at_initial_pathologic_diagnosis","gender","ajcc_pathologic_tumor_stage"]], snps,on="FID")
        #get patients with Stage designations
        full_surv=full_surv[full_surv["ajcc_pathologic_tumor_stage"].str.contains("Stage")]
        full_surv=full_surv[~(full_surv["ajcc_pathologic_tumor_stage"].isin(["Stage X"]))]
        #split A,B,C from stage designations
        full_surv["ajcc_pathologic_tumor_stage"]=full_surv["ajcc_pathologic_tumor_stage"].str.split("A").str[0]
        full_surv["ajcc_pathologic_tumor_stage"]=full_surv["ajcc_pathologic_tumor_stage"].str.split("B").str[0]
        full_surv["ajcc_pathologic_tumor_stage"]=full_surv["ajcc_pathologic_tumor_stage"].str.split("C").str[0]
        full_surv=full_surv.join(pd.get_dummies(full_surv["ajcc_pathologic_tumor_stage"]))
        full_surv["gender"]=full_surv["gender"].map({"MALE":0,"FEMALE":1})
#         full_surv=full_surv[full_surv[surv_type+".time"]<1825]
        full_surv['ajcc_pathologic_tumor_stage'] = full_surv['ajcc_pathologic_tumor_stage'].replace({'Stage 0':0,'Stage I': 1,'Stage II': 2,'Stage III': 3,'Stage IV': 4})
        
    else:
        full_surv=pd.merge(surv[["FID",surv_type,surv_type+".time","age_at_initial_pathologic_diagnosis","gender"]], snps,on="FID")
        full_surv["gender"]=full_surv["gender"].map({"MALE":0,"FEMALE":1})
#         full_surv=full_surv[full_surv[surv_type+".time"]<1825]
    
    return(full_surv, has_tumor)


def main(args):
    
    xgboost_prs_surv_score = pd.read_csv("/cellar/users/sgoudarzi/python_scripts/lasso_xgboost_surv/xgboost_score/all_data_classifier_median/"+args.cancer+"_score.csv")
    sig_snps = pd.read_csv("/cellar/users/sgoudarzi/new_os_cancer_meqtls_sig_pval.csv")
    sig_snps = sig_snps[sig_snps["cancer"]==args.cancer]        
    surv = pd.read_csv("/cellar/users/sgoudarzi/tcga_survival_modified.csv", index_col=0)
    surv = surv[surv["type"]==args.cancer]
    os_all, has_tumor = make_df(args.cancer, surv, "OS", "/cellar/users/sgoudarzi/modeling_prognosis/new_cancer_meqtls/os_kaplanmeier_sig.raw", sig_snps)

    os_all = os_all.dropna()
    os_all = pd.merge(os_all, xgboost_prs_surv_score, on="FID")
    os_all = os_all.rename(columns={args.cancer:"Median_Survival"})
    
    #Check if tumor stage should be used, remove gender for breast and pancreatic cancer
    #Add tumor subtype information for breast cancer as covariates 
    if (has_tumor):
        
        if args.cancer == "BRCA" or args.cancer == "PRAD":
            if args.cancer == "PRAD":
                cox_df = os_all[["PSS", "age_at_initial_pathologic_diagnosis", "ajcc_pathologic_tumor_stage", "OS.time", "Median_Survival"]]
            else:
                cox_df = os_all[["PSS", "age_at_initial_pathologic_diagnosis", "ajcc_pathologic_tumor_stage", "OS.time", "Median_Survival", "breast_carcinoma_estrogen_receptor_status", "breast_carcinoma_progesterone_receptor_status", "lab_proc_her2_neu_immunohistochemistry_receptor_status"]]
        else:
            cox_df = os_all[["PSS", "age_at_initial_pathologic_diagnosis","gender", "ajcc_pathologic_tumor_stage", "OS.time", "Median_Survival"]]
        
        
        
        if args.cancer=="BLCA":
            for i, row in cox_df.iterrows():
                if row["ajcc_pathologic_tumor_stage"] == 1:
                    cox_df.loc[i,"ajcc_pathologic_tumor_stage"] = 2

        #Scale XGBoost Scores
        minmax_scale = MinMaxScaler()
        x = cox_df["PSS"].values.reshape(-1, 1)
        cox_df["PSS"] = minmax_scale.fit_transform(x)

        dummies_stage = pd.get_dummies(cox_df["ajcc_pathologic_tumor_stage"], prefix = 'tumor_stage')
        dummies_cols = list(dummies_stage.columns)
        
        #combine tumor stage 2 with 1 for BRCA, otherwise use tumor stage 1 as the basis for Cox
        if args.cancer == "BLCA":
            dummies_cols.remove("tumor_stage_2")
        else:
            dummies_cols.remove("tumor_stage_1")
        dummies_stage = dummies_stage[dummies_cols]
        cox_df = pd.concat([cox_df, dummies_stage], axis = 1)
        cox_df = cox_df.drop("ajcc_pathologic_tumor_stage", axis = 1)
        cox_df = cox_df.rename(columns={"OS.time":"time", "Median_Survival":"status"})

        #Run CoxPH with PRS and Covariates and OS time as the output
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col = 'time', event_col = 'status')

        plt.subplots(figsize = (12, 7))
        ax = cph.plot()
        ax.set_title(args.cancer+" XGBoost COXPh Hazard Ratio")
        ax.get_figure().savefig("/cellar/users/sgoudarzi/python_scripts/coxph/coxplots/all_data_median_classifier_coxph/"+args.cancer+"_hazardplot.png", bbox_inches='tight')
        ax.get_figure().savefig("/cellar/users/sgoudarzi/python_scripts/coxph/coxplots/all_data_median_classifier_coxph/"+args.cancer+"_hazardplot.pdf", bbox_inches='tight')
        
    else:
            
            
        if args.cancer == "PRAD" or args.cancer=="UCEC" or args.cancer=="OV":
                cox_df = os_all[["PSS", "age_at_initial_pathologic_diagnosis", "OS.time", "Median_Survival"]]
        else:
                cox_df = os_all[["PSS", "age_at_initial_pathologic_diagnosis","gender", "OS.time", "Median_Survival"]]
        
        
        minmax_scale = MinMaxScaler()
        x = cox_df["PSS"].values.reshape(-1, 1)
        cox_df["PSS"] = minmax_scale.fit_transform(x)
        cox_df = cox_df.rename(columns={"OS.time":"time", "Median_Survival":"status"})
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col = 'time', event_col = 'status')
        
        plt.subplots(figsize = (12, 7))
        ax = cph.plot()
        ax.set_title(args.cancer+" XGBoost COXPh Hazard Ratio")
        ax.get_figure().savefig("/cellar/users/sgoudarzi/python_scripts/coxph/coxplots/all_data_median_classifier_coxph/"+args.cancer+"_hazardplot.png", bbox_inches='tight')
        ax.get_figure().savefig("/cellar/users/sgoudarzi/python_scripts/coxph/coxplots/all_data_median_classifier_coxph/"+args.cancer+"_hazardplot.pdf", bbox_inches='tight')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', type=str, help='cancer type')
    args = parser.parse_args()
    main(args)