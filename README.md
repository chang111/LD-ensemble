
## Dependencies

* python 3.6
* [PyTorch 0.4.0](https://pytorch.org/get-started/locally/)
* [XGBoost 0.8](https://pypi.org/project/xgboost/)


## Code

### Demo

In this demo, we developed a total of eight algorithms. Four of them are independent algorithm models and the other four are integrated models:
  1. The LR binary model:
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 7 -C 0.001 -gt LME_Al_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 1 -C 0.0001 -gt LME_Co_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 7 -C 0.01 -gt LME_Ni_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 5 -C 0.01 -gt LME_Ti_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 10 -C 0.01 -gt LME_Le_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 10 -C 0.001 -gt LME_Zi_Close
 * The model performance will be found in `test_result/` folder.
 2.  The LR three classification model:
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 5 -C 0.0001 -gt LME_Al_Close 
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 1 -C 0.0001 -gt LME_Co_Close 
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 10 -C 0.001 -gt LME_Ni_Close
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 1 -C 0.0001 -gt LME_Ti_Close
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 7 -C 0.001 -gt LME_Le_Close 
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 7 -C 0.001 -gt LME_Zi_Close
 3. The XGBoost model:
 4. The DA-LSTM model:
  python code/train/train_da_lstm_online.py -c exp/online_metal.conf -v v5 -s 1 -l 3 -hidden 40 -gt LME > 
 5. The voting model:
  python ijcai_weight_multi.py
 6. The weight model:
  python weight_multi.py
 7. The LR weight model:
  python lr_ensemble.py 
 8. The LD-Ensemble:
  python LD-ensemble.py



