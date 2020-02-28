import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from copy import copy,deepcopy
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from data.load_data import load_data
from model.logistic_regression import LogReg
from utils.transform_data import flatten
from utils.construct_data import rolling_half_year
from utils.log_reg_functions import objective_function, loss_function
from utils.read_data import read_data_NExT
from utils.evaluator import profit_ratio
import warnings
import xgboost as xgb
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import KFold
from utils.version_control_functions import generate_version_params


if __name__ == '__main__':
    desc = 'the logistic regression model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='exp/3d/Co/logistic_regression/v5/LMCADY_v5.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=5,
                        help='steps in the future to be predicted')
    parser.add_argument('-gt', '--ground_truth', help='ground truth column',
                        type=str, default="LME_Co_Spot")
    parser.add_argument('-max_iter','--max_iter',type=int,default=100,
                        help='max number of iterations')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-mout', '--model_save_path', type=str, help='path to save model',
        default='../../exp/log_reg/model'
    )
    parser.add_argument(
        '-l','--lag', type=int, default = 5, help='lag'
    )
    parser.add_argument(
        '-k','--k_folds', type=int, default = 10, help='number of folds to conduct cross validation'
    )
    parser.add_argument(
        '-v','--version', help='version', type = str, default = 'v10'
    )
    parser.add_argument ('-out','--output',type = str, help='output file', default ="../../../Results/results")
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, tune')
    parser.add_argument('-C', '--C', type=float, default=0.001,
                        help='lambda inverse')
    parser.add_argument('-label','--label',type = int, default = None)
    parser.add_argument('-xgb','--xgboost',type = int,help='if you want to train the xgboost you need to inform us of that',default=0)
    args = parser.parse_args()
    if args.ground_truth =='None':
        args.ground_truth = None
    os.chdir(os.path.abspath(sys.path[0]))
    # read data configure file
    with open(os.path.join(sys.path[0],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.ground_truth = args.ground_truth.split(",")
    #print("args.ground_truth is {}".format(args.ground_truth))
    #import os
    #os._exit(0)
    '''
    if args.lag==5 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.8
        learning_rate=0.9
        max_depth=4
        subsample=0.9
    elif args.lag==5 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.9
        learning_rate=0.7
        max_depth=5
        subsample=0.85
    elif args.lag==10 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.9
        learning_rate=0.9
        max_depth=4
        subsample=0.7
    elif args.lag==10 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.8
        learning_rate=0.8
        max_depth=6
        subsample=0.9
    elif args.lag==20 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.7
        learning_rate=0.8
        max_depth=4
        subsample=0.7
    elif args.lag==20 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.8
        learning_rate=0.7
        max_depth=4
        subsample=0.9
    elif args.lag==30 and args.ground_truth[0]=='LME_Ti_Spot':
        gamma=0.7
        learning_rate=0.8
        max_depth=4
        subsample=0.7
    elif args.lag==30 and args.ground_truth[0]=='LME_Co_Spot':
        gamma=0.7
        learning_rate=0.7
        max_depth=4
        subsample=0.9
    '''
    if args.action == 'train':
        comparison = None
        n = 0

        #iterate over list of configurations
        for f in fname_columns:
            lag = args.lag
            temp, stopholder = read_data_NExT(f, "2010-01-06")
            #read data
            if args.source == "NExT":
                data_list, LME_dates = read_data_NExT(f, "2010-01-06")
                time_series = pd.concat(data_list, axis = 1, sort = True)
            elif args.source == "4E":
                from utils.read_data import read_data_v5_4E
                time_series, LME_dates = read_data_v5_4E("2003-11-12")
            
            temp = pd.concat(temp, axis = 1, sort = True)
            columns = temp.columns.values.tolist()
            time_series = time_series[columns]
            # initialize parameters for load data
            length = 5
            split_dates = rolling_half_year("2010-01-01","2020-01-01",length)
            #print(split_dates)
            
            split_dates  =  split_dates[:]
            importance_list = []
            version_params=generate_version_params(args.version)
            if args.label  is not None:
                print("acc label")
                version_params["labelling"] = "v4"
            ans = {"C":[],"ground_truth":args.ground_truth}
            print(split_dates)
            
            for s, split_date in enumerate(split_dates[:-1]):
                #print("the train date is {}".format(split_date[0]))
                #print("the test date is {}".format(split_date[1]))
                horizon = args.steps
                norm_volume = "v1"
                norm_3m_spread = "v1"
                norm_ex = "v1"
                len_ma = 5
                len_update = 30
                tol = 1e-7
                if args.xgboost==1:
                    print(args.xgboost)
                    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':True}
                else:
                    norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                                'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False, 'DALSTM':False}
                final_X_tr = []
                final_y_tr = []
                final_X_va = []
                final_y_va = []
                final_X_te = []
                final_y_te = [] 
                tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                                'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
                ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])

                #load data
                X_tr, y_tr, X_va, y_va, X_te, y_te, norm_check,column_list = load_data(copy(ts),LME_dates,horizon,args.ground_truth,lag,copy(split_date),norm_params,tech_params,version_params)
                print(len(X_tr[0]))
                #post load processing and metal id extension
                """
                X_tr = np.concatenate(X_tr)
                X_tr = X_tr.reshape(len(X_tr),lag*len(column_list[0]))
                y_tr = np.concatenate(y_tr)
                X_va = np.concatenate(X_va)
                X_va = X_va.reshape(len(X_va),lag*len(column_list[0]))
                label = pd.DataFrame(y_va[0],columns=['Label'])
                #label.to_csv(args.ground_truth[0]+str(args.steps)+"_"+split_date[1]+"_lr_"+args.version+"_label.csv")
                y_va = np.concatenate(y_va)
                final_X_tr.append(X_tr)
                final_y_tr.append(y_tr)
                final_X_va.append(X_va)
                final_y_va.append(y_va)
                if args.label is not None:
                    train_y = y_tr > 0
                    val_y = y_va > 0
            

                n+=1
                if args.C not in ans["C"]:
                    ans["C"].append(args.C)
                max_iter = args.max_iter
                pure_LogReg = LogReg(parameters={})
                if split_date[1]+"_acc" not in ans.keys():
                    ans[split_date[1]+"_acc"] = []
                    ans[split_date[1]+"_length"] = []
                if args.label is not None:
                    parameters = {"penalty":"l2", "C":args.C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(f)*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
                    pure_LogReg.train(X_tr,train_y.flatten(), parameters)
                    prediction = pure_LogReg.predict(X_va)
                    #train_prediction = pure_logReg.predict(X_tr)
                    correct = [a == b for a,b in zip(prediction,val_y)]
                    ans[split_date[1]+"_acc"].append(profit_ratio(correct,y_va)[0])
                    ans[split_date[1]+"_length"].append(len(y_va.flatten()))
                else:
                    parameters = {"penalty":"l2", "C":args.C, "solver":"lbfgs", "tol":tol,"max_iter":6*4*len(f)*max_iter, "verbose" : 0,"warm_start": False, "n_jobs": -1}
                    pure_LogReg.train(X_tr,y_tr.flatten(), parameters)
                    #train_prediction = pure_logReg.predict(X_tr)
                    acc= pure_LogReg.test(X_va,y_va.flatten())
                    ans[split_date[1]+"_acc"].append(acc)
                    ans[split_date[1]+"_length"].append(len(y_va.flatten()))
                prob = pure_LogReg.predict_proba(X_va)
                train_prediction = pure_LogReg.predict(X_tr)
                #np.savetxt(args.ground_truth[0]+str(args.steps)+"_"+split_date[1]+"_lr_"+args.version+"_probability.txt",prob)
                np.savetxt(args.ground_truth[0]+str(args.steps)+"_"+split_date[1]+"_lr_"+args.version+"_probability_train.txt",train_prediction)

            pd.DataFrame(ans).to_csv("_".join(["log_reg_online",args.version,str(args.ground_truth[0]),str(args.steps)+".csv"]))"""