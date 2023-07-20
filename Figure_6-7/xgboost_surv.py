import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score,StratifiedKFold
import sys
sys.path.insert(0, "/cellarold/users/mpagadal/Programs/anaconda3/lib/python3.7/site-packages")
from sklearn import linear_model
# from regressors import stats
import statsmodels.api as sm
import statsmodels.stats.multitest as multi
from scipy.stats import mannwhitneyu
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mannwhitneyu
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import argparse
import warnings
warnings.filterwarnings("ignore")

def make_df(surv,surv_type,raw, sig_snps):
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
        full_surv=pd.merge(surv[["FID",surv_type,surv_type+".time", "age_at_initial_pathologic_diagnosis","gender"]], snps,on="FID")
        full_surv["gender"]=full_surv["gender"].map({"MALE":0,"FEMALE":1})
#         full_surv=full_surv[full_surv[surv_type+".time"]<1825]
    
    return(full_surv, has_tumor)

def balancing(data_x, data_y):
    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0)
    columns = X_train.columns
    os_data_X,os_data_y=os.fit_resample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
    os_data_y= pd.DataFrame(data=os_data_y,columns=['Labels'])
    return os_data_X, os_data_y, X_test, y_test

# Create Labels for Median Survival
def create_labels(df, median):
    labels1 = []
    for i, row in df.iterrows():
        if (row["OS.time"]<median):
            labels1.append(1)
        else:
            labels1.append(0)
    df["Labels"] = labels1
    return df

#Only Consider 0 and 1 Genotypes (2 or homozygous allele is considered 1)
def simplify(df):
    for i, row in df.iterrows():
        for col in df:
            if (":" in col):
                if df.at[i, col] == 1 or df.at[i, col] == 2:
                    df.at[i, col] = 1
    return df


def do_LASSO_xgBoost(LASSO_x, LASSO_y, name, cancer_type, control_auc):
    
    #filter using LASSO regression
   
    lasso_filtered_snps = []
    xgboost_importance_snps = dict()
    model = Lasso(alpha=0.001, random_state=10)
    
    model.fit(LASSO_x, LASSO_y)
    
    #Drop features with 0 importance stated by LASSO
    all_data = pd.concat([LASSO_x, LASSO_y], axis=1)
    data_dropped=all_data.drop(LASSO_x.columns.values[model.coef_==0],axis=1)
    df_x_dropped = data_dropped.drop(["Labels"], axis=1)    
    df_y_dropped = all_data["Labels"]
    
    
    #Run a Classifer XGBoost
    
    classifier_xgb = xgboost.XGBRFClassifier(n_estimators =500, random_state = 10,learning_rate=0.1,max_depth=9)
    classifier_xgb.fit(df_x_dropped, df_y_dropped)  
 
    pred_all = classifier_xgb.predict_proba(df_x_dropped)  # test the output by changing values

    # define model evaluation method
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    
    #evaluate model

    scores = cross_val_score(classifier_xgb, df_x_dropped, df_y_dropped, scoring='roc_auc', cv=cv, n_jobs=-1)
    #force scores to be positive
    scores = np.absolute(scores)
    auc_value = np.mean(scores)
    if name == "control":
        return auc_value
    
    if name == "All+control":
        with open("/cellar/users/sgoudarzi/python_scripts/lasso_xgboost_surv/xgboost_score/all_data_classifier_median/auc_values.txt", "a") as file:
            file.write(str(control_auc)+" clinical "+" "+cancer_type+" "+"\n")
            file.write(str(auc_value)+" clinical+snp "+" "+cancer_type+" "+"\n")
            
        if auc_value > control_auc:
            return True
        else:
            return False
    

    xgboost_score = pd.DataFrame()
    xgboost_score["FID"] = list(df_x_dropped.index)
    xgboost_score["PSS"] = pred_all[:, 1]
    xgboost_score[cancer_type] = list(df_y_dropped)
    
    #remove outliers
    Q1,Q3 = np.percentile(xgboost_score["PSS"] , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    xgboost_score = xgboost_score[(xgboost_score["PSS"]>=lower_range) & (xgboost_score["PSS"]<=upper_range)]
    xgboost_score = xgboost_score.reset_index(drop=True)
    
    xgboost_score.to_csv("/cellar/users/sgoudarzi/python_scripts/lasso_xgboost_surv/xgboost_score/all_data_classifier_median/"+cancer_type+"_score.csv")
            
    #Plot the score for 0 and 1 Median Survival Labels
    fig2 = plt.figure(figsize=(12,12))
    sns.boxplot(x=cancer_type,y="PSS",data=xgboost_score)

    cat1 = xgboost_score[xgboost_score[cancer_type]==0]
    cat2 = xgboost_score[xgboost_score[cancer_type]==1]
    disc_score=cat2["PSS"].mean()-cat1["PSS"].mean()
    #print("Difference in PRS: "+str(cat2["LASSO_burden"].median()-cat1["LASSO_burden"].median()))
    plt.title("PSS in "+ cancer_type +" OS, P-value:"+str(stats.mannwhitneyu(cat1["PSS"], cat2["PSS"])[1]), fontsize=19)
    plt.ylabel("PSS for SNPs (p-value<0.05)", fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(rotation=90, fontsize=16)
    plt.xlabel(cancer_type+" Phenotype", fontsize=20)

    plt.savefig("/cellar/users/sgoudarzi/python_scripts/lasso_xgboost_surv/os_sig_classifier_median/xgboost_regression_"+cancer_type+"_OS_sig_burden.png", bbox_inches='tight')
    
    #Plot feature importances
    features = list(df_x_dropped.columns)
    importances = classifier_xgb.feature_importances_
    indices = np.argsort(importances)
    for i in indices:
        xgboost_importance_snps[features[i]] = importances[i]
    
    with open('/cellar/users/sgoudarzi/python_scripts/lasso_xgboost_surv/feature_importance_os_sig_classifier_median/'+cancer_type+'_importances.txt', "w") as file:
        for key in xgboost_importance_snps:
            file.write(key+" "+str(xgboost_importance_snps[key])+"\n")
    
    plt.figure(figsize=(15,20))
    plt.title('Feature Importances for '+cancer_type, fontsize=19)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance', fontsize=16)
    
    plt.savefig("/cellar/users/sgoudarzi/python_scripts/lasso_xgboost_surv/feature_importance_os_sig_classifier_median/feature_importance_os_sig_"+cancer_type+".png", bbox_inches='tight')
    


def main(args):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    
    #read in survival and 156 Cancer-clumped meQTLs
    sig_snps = pd.read_csv("../meQTL_Data/new_os_cancer_meqtls_sig_pval.csv")
    sig_snps = sig_snps[sig_snps["cancer"]==args.cancer]     
    #TCGA Survival
    surv = pd.read_csv("/cellar/users/mpagadal/resources/from-labmembers/andrea/Liu2018.TCGA_survival.csv", index_col=0)
    surv = surv[surv["type"]==args.cancer]
    os_all, has_tumor = make_df(surv, "OS", "/cellar/users/sgoudarzi/modeling_prognosis/new_cancer_meqtls/os_kaplanmeier_sig.raw", sig_snps)
    os_all = os_all[os_all['age_at_initial_pathologic_diagnosis'].notna()]
    os_all = os_all[os_all['OS'].notna()]
    os_all = os_all[os_all['OS.time'].notna()]
    
    #Don't run analysis in tumors with less than 100 implicated SNPs and less than 5 significant SNPs
    if len(os_all)<100 and len(sig_snps) < 5:
        return
    
    #Create Survival Median as Label for XGBoost Classifier
    os_all = create_labels(os_all, np.median(os_all["OS.time"]))

    #Check if tumor stage should be used or not
    if (has_tumor):
        os_all_X = os_all[['age_at_initial_pathologic_diagnosis','gender', 'ajcc_pathologic_tumor_stage']]
        os_all_X['age_at_initial_pathologic_diagnosis'] = stats.zscore(np.array(os_all_X['age_at_initial_pathologic_diagnosis']))
        os_all_y = os_all[['Labels']]
        test1 = pd.get_dummies(os_all_X["ajcc_pathologic_tumor_stage"])
        test1 = test1.rename(columns={0:"ajcc_pathologic_tumor_stage_0", 1:"ajcc_pathologic_tumor_stage_1", 2:"ajcc_pathologic_tumor_stage_2",   3:"ajcc_pathologic_tumor_stage_3", 4:"ajcc_pathologic_tumor_stage_4"})
        os_all_X = pd.concat([os_all_X, test1], axis=1)
        #print(os_active_X.columns)
        del os_all_X["ajcc_pathologic_tumor_stage"]
    else:
        os_all_X = os_all[['age_at_initial_pathologic_diagnosis','gender']]
        os_all_X['age_at_initial_pathologic_diagnosis'] = stats.zscore(np.array(os_all_X['age_at_initial_pathologic_diagnosis']))
        os_all_y = os_all[['Labels']]
        
     
    #Run XGBoost on the Control with Only Clinical Covariates
    control_auc = do_LASSO_xgBoost(os_all_X, os_all_y, "control", args.cancer, 0)
    
    #Add SNPs to the Dataset
    os_all = simplify(os_all)
    os_all_X_no_covariates = pd.DataFrame()
    
    #Remove SNP with less than 0.01 MAF value and Z-Score Genotypes
    for col in os_all:
        if ":" in col:
            if args.cancer == "PAAD" and "2:209220238" in col:
                continue
            os_all_X[col] = stats.zscore(np.array(os_all[col]))
            os_all_X_no_covariates[col] = stats.zscore(np.array(os_all[col]))


    os_all_y["FID"] = list(os_all["FID"])
    os_all_y = os_all_y.set_index("FID")
    os_all_X["FID"] = list(os_all["FID"])
    os_all_X = os_all_X.set_index("FID")
    os_all_X_no_covariates["FID"] = list(os_all["FID"])
    os_all_X_no_covariates = os_all_X_no_covariates.set_index("FID")
    
    cont = do_LASSO_xgBoost(os_all_X, os_all_y, "All+control", args.cancer, control_auc)
    
    #if AUC of SNP+clinical > clinical, then proceed with XGBoost with Just SNP
    if cont:
        do_LASSO_xgBoost(os_all_X_no_covariates, os_all_y, "All", args.cancer, 0)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', type=str, help='cancer type')
    args = parser.parse_args()
    main(args)