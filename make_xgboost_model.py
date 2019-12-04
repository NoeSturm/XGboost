import numpy as np
from scipy.io import mmread
import scipy.sparse
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import os,sys
import pandas as pd
import pickle

os.chdir("/projects/cc/kxsw653/ExCAPE-ML_v5/ML_models_results/results_v5_xgboost_ecfp")

if len(sys.argv)!=7: 
    print("Usage: {} [target_idx] [obj] [lr] [scale_pos_w] [n_Estim] [max_depth]".format(sys.argv[0]) )
    quit()

target_idx       = int(sys.argv[1])
objective        = sys.argv[2]
learning_rate    = float(sys.argv[3])
scale_pos_weight = int(sys.argv[4])
n_estim          = int(sys.argv[5])
max_depth        = int(sys.argv[6])


test_file  = "data/inner_folds/test_s1_c1_i1_al6.mtx"
feat_file  = "data/side_info/ecfp6_counts_var005.mtx"
train_file = "data/train_s1_c1_al6.mtx"
print("Load train data")
train_data = mmread(train_file).tocsc(copy=False)[:,target_idx]
print("train size: {}".format(train_data.shape))

#print("load test data")
#test_data  = mmread(test_file).tocsc(copy=False)[:,target_idx]

print("load features")
features = mmread(feat_file).tocsr(copy=False)

print("slice the matirx for target: {}".format(target_idx))
cmp_idx_train = train_data.indices
X_train = features[cmp_idx_train,:]
y_train = train_data.data
print("features size:{}".format(X_train.shape))
#cmp_idx_test = test_data.indices
#X_test = features[cmp_idx_test,:]
#y_test = test_data.data


print("Make XGBoost Classif model with :")
print("Param_set: object:{} - lr:{} - scale_pos_weight:{} - n_estim:{} - max_depth:{}".format(objective,learning_rate,scale_pos_weight,n_estim,max_depth))
model = XGBClassifier(booster='gbtree', learning_rate=learning_rate, n_estimators=n_estim, max_depth=max_depth, objective=objective, scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)

#print("Make predictions:" )
#y_pred = model.predict(X_test)
#
#hyperparam_str = "C-{}_penalty-{}_loss-{}_dual-{}".format(C,penalty,loss,dual)
#predictions = pd.DataFrame()
#predictions['y_pred'] = y_pred
#predictions['y_test'] = y_test
#predictions['hyperparam'] = hyperparam_str
#predictions['target'] = target_idx

WORKDIR="/projects/cc/kxsw653/ExCAPE-ML_v5/ML_models_results/results_v5_xgboost_ecfp"

#print("predictions saved in {}/nested-cv-predictions/predictions-1-1-1-6-{}.pickle".format(WORKDIR, target_idx))
#predictions.to_pickle("{}/nested-cv-predictions/predictions-1-1-1-6-{}.pickle".format(WORKDIR,target_idx))

print("save model in {}/full-scale-models/xgboost_full_scale_best_ofold_mean/model-full-{}.pickle".format(WORKDIR, target_idx))
f = open("{}/full-scale-models/xgboost_full_scale_best_ofold_mean/model-1-full-6-{}.pickle".format(WORKDIR, target_idx), "wb")
pickle.dump(model, f)
f.close()

#print("compute perf")
#recall = recall_score(predictions['y_test'].values, predictions['y_pred'].values)
#precision=precision_score(predictions['y_test'].values, predictions['y_pred'].values)
#kappa=cohen_kappa_score(predictions['y_test'].values, predictions['y_pred'].values)
#rocauc=roc_auc_score(predictions['y_test'].values, predictions['y_pred'].values)
#scores = "{}\t{}\t{}\t{}\t{}".format(hyperparam_str, rocauc, recall, precision, kappa)

#print("save scores in {}/nested-cv-perf-scores.tmp/scores-1-1-1-6-{}-{}.csv".format(WORKDIR, target_idx,hyperparam_str))
#f = open("{}/nested-cv-perf-scores.tmp/scores-1-1-1-6-{}-{}.csv".format(WORKDIR, target_idx,hyperparam_str), "w")
#f.write(scores)
#f.close()


