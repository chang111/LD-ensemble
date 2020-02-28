import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from sklearn import metrics
import math
import argparse
from copy import copy
from sklearn import tree
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn import tree

if __name__ == '__main__':
    for horizon in [1]:
        for single_window in [10]:
            for ground_truth in ["LME_Co_Close","LME_Al_Close","LME_Ni_Close","LME_Ti_Close","LME_Zi_Close","LME_Le_Close"]:
                for date in ["2015-01-01","2015-07-01","2016-01-01","2016-07-01","2017-01-01","2017-07-01","2018-01-01","2018-07-01","2019-01-01"]:
                    lr_error_v5 = []
                    xgb_error_v5 = []
                    da_error_v5 = []
                    final_list_lr_ensemble_train = []
                    y_va = pd.read_csv("data/Label/"+ground_truth+str(horizon)+"_"+date+"_lr_v1"+"_label"+".csv")
                    label = y_va['Label'].values.tolist()
                    label = [-1 if item==0 else 1 for item in label]
                    label = np.array(label).reshape(-1,1).tolist()
                    lr_v5 = np.loadtxt("data/lr_test_result/"+ground_truth+str(horizon)+"_"+date+"_lr_v5_probability.txt")
                    xgboost_v5 = np.loadtxt("data/xgboost_test_result/"+ground_truth+"_"+"h"+str(horizon)+"_"+date+"_xgboost_v5.txt")
                    DALSTM_v5=np.loadtxt("data/DALSTM_test_result/"+'LME'+str(horizon)+"_"+date+"_ALSTM_"+"v5"+"_result.txt")
                    ground_truths_list = ["LME_Co_Close","LME_Al_Close","LME_Ni_Close","LME_Ti_Close","LME_Zi_Close","LME_Le_Close"]
                    lr_three = np.loadtxt("data/lr_three_classifier/"+ground_truth+"1"+"_"+date+"_lr_ex2_"+"v5_ex2"+"_probability.txt")                  
                    final_list_v5_lr = []
                    final_list_v5_xgb = []
                    final_list_lr_ensemble_train_new = []
                    for j in range(len(lr_v5)):
                        if lr_v5[j]>0.5:
                            final_list_v5_lr.append(1)
                        else:
                            final_list_v5_lr.append(0)
                    #lr_v1.append()
                    final_list_v5_lr = [-1 if item==0 else 1 for item in final_list_v5_lr]
                    for j in range(len(xgboost_v5)):
                        count_1 = 0
                        count_0 = 0
                        for item in xgboost_v5[j]:
                            if item>0.5:
                                count_1+=1
                            else:
                                count_0+=1
                        if count_1>count_0:
                            final_list_v5_xgb.append(1)
                        else:
                            final_list_v5_xgb.append(0)                 
                    final_list_v5_xgb = [-1 if item==0 else 1 for item in final_list_v5_xgb]
                    length = len(DALSTM_v5)
                    metal = 6
                    day = int(length/6)
                    #print(day)
                    new_DALSTM_v5 = []
                    for i in range(metal):
                        new_list = []
                        for j in range(day):
                            new_list.append(copy(DALSTM_v5[j*6+i]))
                        new_DALSTM_v5.append(new_list)                                             
                    if ground_truth == ground_truths_list[0]:
                        DALSTM_v5=copy(new_DALSTM_v5[0])
                    elif ground_truth == ground_truths_list[1]:
                        DALSTM_v5=copy(new_DALSTM_v5[1])
                    elif ground_truth == ground_truths_list[2]:
                        DALSTM_v5=copy(new_DALSTM_v5[2])
                    elif ground_truth == ground_truths_list[3]:
                        DALSTM_v5=copy(new_DALSTM_v5[3])
                    elif ground_truth == ground_truths_list[4]:
                        DALSTM_v5=copy(new_DALSTM_v5[4])
                    elif ground_truth == ground_truths_list[5]:
                        DALSTM_v5=copy(new_DALSTM_v5[5])
                    final_list_v5_DALSTM = DALSTM_v5
                    final_list_v5_DALSTM = [-1 if item==0 else 1 for item in final_list_v5_DALSTM]
                    results = []
                    final_list_1 = []
                    for i in range(len(final_list_v5_lr)):

                        if label[i]!=final_list_v5_lr[i]:
                            lr_error_v5.append(1)
                        else:
                            lr_error_v5.append(0)                   

                        if label[i]!=final_list_v5_xgb[i]:
                            xgb_error_v5.append(1)
                        else:
                            xgb_error_v5.append(0)

                        if label[i]!=final_list_v5_DALSTM[i]:
                            da_error_v5.append(1)
                        else:
                            da_error_v5.append(0)

                        if final_list_v5_lr[i]+final_list_v5_xgb[i]+final_list_v5_DALSTM[i]>=1:
                            results.append(1)
                            if i < horizon:
                                final_list_1.append(1)
                                final_list_lr_ensemble_train_new.append(1)
                        else:
                            results.append(-1)
                            if i < horizon:
                                final_list_1.append(-1)
                                final_list_lr_ensemble_train_new.append(-1)                              
                    #print("the voting result is {}".format(metrics.accuracy_score(label, results)))
                    window = 1
                    length = 0
                    X_tr = []
                    y_tr = []
                    for i in range(horizon,len(label)):
                        va = []
                        error_lr_v5 = np.sum(lr_error_v5[length:length+window])+1e-06
                        
                        error_xgb_v5 = np.sum(xgb_error_v5[length:length+window])+1e-06

                        error_da_v5 = np.sum(da_error_v5[length:length+window])+1e-06                                        
                        result = 0
                        fenmu =1/error_lr_v5+1/error_xgb_v5+1/error_da_v5
                        weight_lr_v5 = float(1/error_lr_v5)/fenmu
                        result+=weight_lr_v5*final_list_v5_lr[i]
                        va.append(weight_lr_v5*final_list_v5_lr[i])
                        weight_xgb_v5 = float(1/error_xgb_v5)/fenmu
                        result+=weight_xgb_v5*final_list_v5_xgb[i]
                        va.append(weight_xgb_v5*final_list_v5_xgb[i])
                        weight_da_v5 = float(1/error_da_v5)/fenmu
                        result+=weight_da_v5*final_list_v5_DALSTM[i]
                        va.append(weight_da_v5*final_list_v5_DALSTM[i])
                        X_tr.append(va)
                        y_tr.append(label[i])                                                   
                        if result>0:
                            final_list_1.append(1)
                        else:
                            final_list_1.append(-1)
                        if window==single_window:
                            length+=1
                        else:
                            window+=1
                    # add three classifier to the feature
                    for i in range(len(label)-1):
                        X_tr[i].append(copy(lr_three[i+1]))
                    model = LogisticRegression(C=0.2,penalty="l2",verbose=0,fit_intercept=False,solver="lbfgs",max_iter = 6400)
                    model = model.fit(X_tr[:119], y_tr[:119])
                    final_list_lr_ensemble_train_new += model.predict(X_tr[:119]).tolist()
                    final_list_lr_ensemble_train = copy(final_list_lr_ensemble_train)+final_list_lr_ensemble_train_new
                    final_list_lr_ensemble = model.predict(X_tr[119:])
                    print("the length of the y_test is {}".format(len(label[120:])))
                    print("the weight ensebmle for weight voting beta precision is {}".format(metrics.accuracy_score(label[120:], final_list_lr_ensemble)))
                    print("the horizon is {}".format(horizon))
                    print("the window size is {}".format(single_window))
                    print("the metal is {}".format(ground_truth))
                    print("the test date is {}".format(date))