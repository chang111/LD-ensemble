import os, sys, time, random
import argparse
import torch
import numpy as np
import json
from copy import copy
from torch import nn
from torch.autograd import Variable
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
from model.DALSTM import AttnEncoder, AttnDecoder
from data.load_data import load_data
from model.model_embedding import MultiHeadAttention, attention, bilstm
from utils.construct_data import rolling_half_year
from utils.version_control_functions import generate_version_params
from torch import optim
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
import psutil
import pandas as pd

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

thresh = 0.0

def memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid=pid)
    memory_use = py.memory_info()[0]/2.**30
    print('memory useage:', memory_use)

class Trainer:
 
    def __init__(self, hidden_state, time_step, split_date, lr, version,embedding_size, lambd,
                            X_train, y_train, y_seq_train,
                            X_test, y_test, y_seq_test,
                            num_case):
        #self.dataset = Dataset(time_step, split, 0, version)
        self.X_train = X_train
        self.y_train = y_train
        self.y_seq_train = y_seq_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_seq_test = y_seq_test
        self.feature_size = len(X_train[0][0])
        self.case_size = num_case
        self.encoder = AttnEncoder(input_size=self.feature_size, hidden_size=hidden_state, time_step=time_step)
        self.decoder = AttnDecoder(code_hidden_size=hidden_state, hidden_size=hidden_state, time_step=time_step, case_number=self.case_size, embedding_size=embedding_size)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr)
        self.loss_func = nn.MSELoss()
        """
        old code
        """
        #self.train_size, self.test_size = self.dataset.get_size()
        """
        new code to fit the new data set
        """
        self.train_size = len(X_train)
        self.test_size = len(X_test) 
        self.split = split
        self.lr = lr
        self.window_size = time_step
        self.hidden_state = hidden_state
        self.embedding_size = embedding_size
        self.version = version
        #self.case_size = num_case
        """
        the old code
        """
        #self.train_day, self.test_day = self.dataset.get_day_size()
        self.train_day = int(self.train_size/self.case_size)
        self.test_day = int(self.test_size/self.case_size)

       

    def train_minibatch(self, num_epochs, split_date, interval,h,metal):
        #start = time.time()
        """
        old code
        """
        #x_train, y_train, y_seq_train = self.dataset.get_train_set()
        x_train = self.X_train
        y_train = self.y_train
        y_seq_train = self.y_seq_train
        y_train_class = np.array([1 if ele>thresh else 0 for ele in y_train])
        #print(y_train)
        #print(y_train_class)
        train_loss_list = []
        test_loss_list = []
        
        """
        old code
        """
        #x_test, y_test, y_seq_test = self.dataset.get_test_set()
        x_test = self.X_test
        y_test = self.y_test
        y_seq_test = self.y_seq_test
        y_test_class = np.array([1 if ele>thresh else 0 for ele in y_test])
        end = time.time()
        print("data load is done")

        train_prediction = []
        test_prediction = []
        train_acc_list = []
        test_acc_list = []
        best_acc = 0
        for epoch in range(num_epochs):
            current_train_pred = []
            current_test_pred = []
            
            '''start train'''
            self.encoder.train()
            self.decoder.train()
            
            start = time.time()
            loss_sum = 0
            
            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            
            for day in range(self.train_day):
                start_pos = day*self.case_size
                var_x = self.to_variable(x_train[start_pos:start_pos+self.case_size])
                var_y = self.to_variable(y_train[start_pos:start_pos+self.case_size])
                var_y_seq = self.to_variable(y_seq_train[start_pos:start_pos+self.case_size])
                var_x_id = torch.LongTensor(list(range(self.case_size)))
                code = self.encoder(var_x)
                y_res = self.decoder(code, var_y_seq, var_x_id)
                var_y = var_y.view(-1,1)
                loss = self.loss_func(y_res, var_y)
                loss = loss/5
                loss.backward()

                if (day+1)%5==0 or day==self.train_day-1:
                    self.encoder_optim.step()
                    self.decoder_optim.step()
                    self.encoder_optim.zero_grad()
                    self.decoder_optim.zero_grad()
                
                loss_sum += loss.detach()*5

            end = time.time()
            print('epoch [{}] finished, with time: {}'.format(epoch, end-start))
            
            #memory_usage()
            
            '''start evaluate'''
            self.encoder.eval()
            self.decoder.eval()
            start = time.time()
            loss_sum = 0
            for day in range(self.train_day):

                start_pos = day*self.case_size
                var_x_train = self.to_variable(x_train[start_pos:start_pos+self.case_size])
                var_y_train = self.to_variable(y_train[start_pos:start_pos+self.case_size])
                var_y_train_seq = self.to_variable(y_seq_train[start_pos:start_pos+self.case_size])
                var_x_id = torch.LongTensor(list(range(self.case_size)))

                var_y_train = var_y_train.view(-1,1)
                train_code = self.encoder(var_x_train)
                train_y_res = self.decoder(train_code,var_y_train_seq,var_x_id)
                loss = self.loss_func(train_y_res, var_y_train)
                loss_sum += loss.detach()
                current_train_pred += list(train_y_res.detach().view(-1,))
                
            current_train_class = [1 if ele>thresh else 0 for ele in current_train_pred]
            
            train_loss = loss_sum/self.train_day
            train_loss_list.append(float(train_loss))
            
            train_acc = accuracy_score(y_train_class, current_train_class)
            train_acc_list.append(train_acc)
            
            end = time.time()
            print('the average train loss is: {}, accuracy is {} with time: {}'.format(train_loss, train_acc,end-start))
            
            loss_sum = 0
            start = time.time()
            for day in range(self.test_day):
                
                start_pos = day*self.case_size
                var_x_test = self.to_variable(x_test[start_pos:start_pos+self.case_size])
                var_y_test = self.to_variable(y_test[start_pos:start_pos+self.case_size])
                var_y_test_seq = self.to_variable(y_seq_test[start_pos:start_pos+self.case_size])
                var_x_test_id = torch.LongTensor(list(range(self.case_size)))
                var_y_test = var_y_test.view(-1,1)
                test_code = self.encoder(var_x_test)
                test_y_res = self.decoder(test_code, var_y_test_seq, var_x_test_id)
                loss = self.loss_func(test_y_res, var_y_test)
                loss_sum += loss.detach()
                current_test_pred += list(test_y_res.detach().view(-1,))
            
            current_test_class = [1 if ele>thresh else 0 for ele in current_test_pred]
            
            test_loss = loss_sum/self.test_day
            test_loss_list.append(float(test_loss))
            
            test_acc = accuracy_score(y_test_class, current_test_class)
            test_acc_list.append(test_acc)
            
            end = time.time()
            print('the average validation loss is {}, accurary is {}, with time: {}'.format(test_loss, test_acc,end-start))

            if (epoch+1)%10 == 0:
                current_train_pred = np.array(current_train_pred).reshape(self.case_size,-1)
                current_test_pred = np.array(current_test_pred).reshape(self.case_size,-1)

                train_prediction.append([list(ele) for ele in current_train_pred])
                test_prediction.append([list(ele) for ele in current_test_pred])

            '''now we don't save models'''

            if test_acc > best_acc:
                torch.save(self.encoder.state_dict(), 'encoder-'+ '-date-' + split_date[1] + '-lag-' + str(self.window_size) + '-lr-' + str(self.lr) + '-hidden-' + str(hidden_state) + '-split-' + str(split) + '-version-'+ str(self.version) + "-horizon-"+ str(h) +"-metal-"+metal+'.model')
                torch.save(self.decoder.state_dict(), 'decoder-'+ '-date-' + split_date[1] + '-lag-' + str(self.window_size) + '-lr-' + str(self.lr) + '-hidden-' + str(hidden_state) + '-split-' + str(split) + '-version-'+ str(self.version) + "-horizon-"+ str(h) +"-metal-"+metal+'.model')
                best_acc = test_acc

        #out_pred_train = pd.DataFrame()
        #out_pred_test = pd.DataFrame()
        out_loss = pd.DataFrame()
        out_loss['train_loss'] = train_loss_list
        out_loss['test_loss'] = test_loss_list
        out_loss['train_acc'] = train_acc_list
        out_loss['test_acc'] = test_acc_list
        
        return out_loss

    def test(self, x, y, y_seq):
        #x_test, y_test, y_seq_test = self.dataset.get_test_set()
        y_pred_test = self.predict(x, y, y_seq)
        current_pred_class = [1 if ele>thresh else 0 for ele in y_pred_test]
        current_test_class = [1 if ele>thresh else 0 for ele in y]
        print("the test accuracy: ",accuracy_score(current_test_class,current_pred_class))
    """
    def predict(self, x, y, y_seq):
        y_pred = np.zeros(x.shape[0])
        i = 0
        while (i < x.shape[0]):
            batch_end = i + batch_size
            if batch_end > x.shape[0]:
                batch_end = x.shape[0]
            var_x_input = self.to_variable(x[i: batch_end])
            var_y_input = self.to_variable(y_seq[i: batch_end])
            if var_x_input.dim() == 2:
                var_x_input = var_x_input.unsqueeze(2)
            var_x_test_id = torch.LongTensor(list(range(self.case_size)))
            code = self.encoder(var_x_input)
            y_res = self.decoder(code, var_y_input, var_x_test_id)
            for j in range(i, batch_end):
                y_pred[j] = y_res[j - i, -1]
            i = batch_end
        return y_pred
    """
    def predict(self, x, y, y_seq):
        y_pred = np.zeros(x.shape[0])
        day = int(len(x)/self.case_size)
        for day in range(day):
            start_pos = day*self.case_size
            var_x = self.to_variable(x[start_pos:start_pos+self.case_size])
            var_y = self.to_variable(y[start_pos:start_pos+self.case_size])
            var_y_seq = self.to_variable(y_seq[start_pos:start_pos+self.case_size])
            var_x_id = torch.LongTensor(list(range(self.case_size)))
            code = self.encoder(var_x)
            y_res = self.decoder(code, var_y_seq, var_x_id)
            y_pred[start_pos:start_pos+self.case_size]=copy(list(y_res.detach().view(-1,)))                
        return y_pred    
    def load_model(self, encoder_path, decoder_path):
        # keep the model (return a dict)
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())



def getArgParser():
    parser = argparse.ArgumentParser(description='Train the dual-stage attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=10,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=1024,
        help='the mini-batch size')
    parser.add_argument(
        '-split', '--split', type=float, default=0.9,
        help='the split ratio of validation set')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-lrate', '--lrate', type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='the model name(after encoder/decoder)'
    )
    parser.add_argument(
        '-hidden','--hidden_state',type=int, default=50,
        help='number of hidden_state of encoder/decoder'
    )
    parser.add_argument(
        '-embed','--embedding_size', type=int, default=10,
        help='the size of embedding layer'
    )
    parser.add_argument(
        '-lambd','--lambd', type=float,default=0,
        help='the weight of classfication loss'
    )
    parser.add_argument(
        '-savep','--save_prediction',type=bool, default=1,
        help='whether to save prediction results'
    )
    parser.add_argument(
        '-savel','--save_loss',type=bool, default=1,
        help='whether to save loss results'
    )
    parser.add_argument(
        '--data_configure_file', '-c', type=str,
        help='configure file of the features to be read',
        default='exp/online_metal_v5_v7.conf'
    )
    parser.add_argument('-s','--steps',type=int,default=1,
                        help='steps in the future to be predicted')
    parser.add_argument(
        '-sou','--source', help='source of data', type = str, default = "NExT"
    )
    parser.add_argument(
        '-l','--lag', type=int, default = 5, help='lag'
    )
    parser.add_argument(
        '-v','--version', help='version', type = str, default = 'v7'
    )
    parser.add_argument('-gt', '--gt', help='ground truth column',
                        type=str, default="LME")
    parser.add_argument('-label','--label',type = int, default = None)                                                
    return parser


if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = args.batch
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    mname = args.model
    window_size = args.lag
    hidden_state = args.hidden_state
    version = args.version
    lambd = args.lambd
    save_prediction = args.save_prediction
    save_loss = args.save_loss
    embedding_size = args.embedding_size
    
    if version == -2:
        thresh = -0.04344493
    os.chdir(os.path.abspath(sys.path[1]))
    
    # read data configure file
    with open(os.path.join(sys.path[1],args.data_configure_file)) as fin:
        fname_columns = json.load(fin)
    args.gt = args.gt.split(",")

    comparison = None
    n = 0
    
    #iterate over list of configurations
    for f in fname_columns:
        lag = args.lag
        
        #read data
        if args.source == "NExT":
            from utils.read_data import read_data_NExT
            data_list, LME_dates = read_data_NExT(f, "2010-01-06")
            time_series = pd.concat(data_list, axis = 1, sort = True)
        elif args.source == "4E":
            from utils.read_data import read_data_v5_4E
            time_series, LME_dates = read_data_v5_4E("2003-11-12")
        temp, stopholder = read_data_NExT(f, "2010-01-06")
        temp = pd.concat(temp, axis = 1, sort = True)
        columns = temp.columns.values.tolist()
        time_series = time_series[columns]
        # initialize parameters for load data
        length = 5
        split_dates = rolling_half_year("2010-01-01","2018-01-01",length)
        #split_dates = rolling_half_year("2010-01-01","2019-12-31",length)
        #print(split_dates)
        
        split_dates  =  split_dates[:]
        importance_list = []
        version_params=generate_version_params(args.version)
        if args.label  is not None:
            print("acc label")
            version_params["labelling"] = "v4"
        ans = {"C":[]}
        print(split_dates)
        
        for s, split_date in enumerate(split_dates[:-1]):
            #print("the train date is {}".format(split_date[0]))
            #print("the test date is {}".format(split_date[1]))
            
            #generate parameters for load data
            horizon = args.steps
            norm_volume = "v1"
            norm_3m_spread = "v1"
            norm_ex = "v1"
            len_ma = 5
            len_update = 30
            tol = 1e-7
            norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                        'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False}
            final_X_tr = []
            final_y_tr = []
            final_X_va = []
            final_y_va = []
            final_X_te = []
            final_y_te = [] 
            tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                            'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}
            ts = copy(time_series.loc[split_date[0]:split_dates[s+1][2]])
            # ,mklprint(args.gt)
            if args.gt[0] == "LME":
                ground_truths_list = ["LME_Co_Close","LME_Al_Close","LME_Ni_Close","LME_Ti_Close","LME_Zi_Close","LME_Le_Close"]
            else:
                ground_truths_list = ['China_Corn_Close','China_Cotton_Close','China_Soybeanoil_Close','China_Soybeanmeal_Close','China_Soybeantwo_Close','China_Sugar_Close']
            #load data
            train_X = []
            train_y = []
            train_y_seq = []
            test_X = []
            test_y = []
            test_y_seq = []
            for ground_truth in ground_truths_list:
                print(ground_truth)
                norm_params = {'vol_norm':norm_volume,'ex_spread_norm':norm_ex,'spot_spread_norm':norm_3m_spread,
                            'len_ma':len_ma,'len_update':len_update,'both':3,'strength':0.01,'xgboost':False, 'DALSTM':True}
                tech_params = {'strength':0.01,'both':3,'Win_VSD':[10,20,30,40,50,60],'Win_EMA':12,'Win_Bollinger':22,
                                                'Fast':12,'Slow':26,'Win_NATR':10,'Win_VBM':22,'acc_initial':0.02,'acc_maximum':0.2}                
                #print(tech_params)
                X_tr, y_tr, y_tr_seq, X_va, y_va, y_va_seq, X_te, y_te, norm_params,column_list = load_data(copy(ts),copy(LME_dates),copy(horizon),[ground_truth],copy(lag),copy(split_date),copy(norm_params),copy(tech_params),copy(version_params))    
                train_X.append(X_tr[0])
                train_y.append(y_tr[0])
                train_y_seq.append(y_tr_seq[0])
                test_X.append(X_va[0])
                test_y.append(y_va[0])
                test_y_seq.append(y_va_seq[0])
            #print(train_X[0])
            #print(len(test_X))
            #print(len(test_X[0]))
            #print(len(test_X[1]))
            #print(len(test_X[2]))
            #print(len(test_X[3]))
            #print(len(test_X[4]))
            #print(len(test_X[5]))
            #print(len(train_X[0]))
            new_train_X = []
            val_X = []
            new_train_y = []
            val_y = []
            new_train_y_seq = []
            val_y_seq = []
            train_length = int(len(train_X[0])*split)
            #print(train_length)
            val_length = int(len(train_X[0])-train_length)
            test_length = len(test_X[0])
            for i in range(len(train_X)):
                new_train_X.append(copy(train_X[i][:train_length]))
                val_X.append(copy(train_X[i][train_length:]))
                new_train_y.append(copy(train_y[i][:train_length]))
                val_y.append(copy(train_y[i][train_length:]))
                new_train_y_seq.append(copy(train_y_seq[i][:train_length]))
                val_y_seq.append(copy(train_y_seq[i][train_length:]))
            train_X = new_train_X
            #new_train_y = copy(train_y[:,:train_length])
            #val_y = copy(train_y[:,train_length:])
            train_y = new_train_y
            #new_train_y_seq = copy(train_y_seq[:,:train_length])
            #val_y_seq = copy(train_y_seq[:,train_length:])
            train_y_seq = new_train_y_seq
            #print(length)
            #print(len(train_X[0][0][0]))
            #print()
            #print(len(val_X))
            #print(len(train_X[0]))
            #print()
            X_train = []
            y_train = []
            y_seq_train = []
            X_val = []
            y_val = []
            y_seq_val =[]
            X_test = []
            y_test = []
            y_seq_test = []
            for i in range(train_length):
                for j in range(len(ground_truths_list)):
                    X_train.append(np.array(train_X[j][i]))
                    y_train.append(np.array(train_y[j][i]))
                    y_seq_train.append(np.array(train_y_seq[j][i]))
            for i in range(val_length):
                for j in range(len(ground_truths_list)):
                    X_val.append(np.array(val_X[j][i]))
                    y_val.append(np.array(val_y[j][i]))
                    y_seq_val.append(np.array(val_y_seq[j][i]))            
            for i in range(test_length):
                for j in range(len(ground_truths_list)):
                    X_test.append(np.array(test_X[j][i]))
                    y_test.append(np.array(test_y[j][i]))
                    y_seq_test.append(np.array(test_y_seq[j][i]))
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            y_seq_train = np.array(y_seq_train)
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            y_seq_val = np.array(y_seq_val)            
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            y_seq_test = np.array(y_seq_test)
            num_case = len(ground_truths_list)
            #print(len(X_train[0]))
            print(split_date[1])
            trainer = Trainer(hidden_state, window_size, split_date, lr, version,embedding_size, lambd,
                X_train, y_train, y_seq_train,
                X_val, y_val, y_seq_val,
                num_case)
            if not test:
                out_loss = trainer.train_minibatch(num_epochs, split_date, interval,horizon,args.gt[0])
                #if save_prediction:
                #    out_train_pred.to_csv("results_pred/PredResult-date-"+split_date[1]+"-lr-"+str(lr)+"-window_size-"+str(window_size)+"-hidden_state-"+str(hidden_state)+"-embedding_size-"+str(embedding_size)+"-data_version-"+str(version)+"-"+"train.csv")
                #    out_test_pred.to_csv("results_pred/PredResult-date-"+split_date[1]+"-lr-"+str(lr)+"-window_size-"+str(window_size)+"-hidden_state-"+str(hidden_state)+"-embedding_size-"+str(embedding_size)+"-data_version-"+str(version)+"-"+"test.csv")
                #    print("Training prediction and test prediction saved! ")
                #if save_loss:
                #    out_loss.to_csv("results_loss/LossResult-date-"+split_date[1]+"-lr-"+str(lr)+"-window_size-"+str(window_size)+"-hidden_state-"+str(hidden_state)+"-embedding_size-"+str(embedding_size)+"-data_version-"+str(version)+"-"+comment+".csv")
                #print(out_loss)
                encoder_name = 'encoder-'+ '-date-' + split_date[1] + '-lag-' + str(window_size) + '-lr-' + str(lr) + '-hidden-' + str(hidden_state) + '-split-' + str(split) + '-version-'+ str(version) + "-horizon-"+ str(horizon) +"-metal-"+args.gt[0]+'.model'
                decoder_name = 'decoder-'+ '-date-' + split_date[1] + '-lag-' + str(window_size) + '-lr-' + str(lr) + '-hidden-' + str(hidden_state) + '-split-' + str(split) + '-version-'+ str(version) + "-horizon-"+ str(horizon) +"-metal-"+args.gt[0]+'.model'
                trainer.load_model(encoder_name, decoder_name)
                trainer.test(X_test, y_test, y_seq_test)
                #print()



