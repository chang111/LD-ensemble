
## Dependencies

* python 3.6
* [PyTorch 0.4.0](https://pytorch.org/get-started/locally/)
* [XGBoost 0.8](https://pypi.org/project/xgboost/)


## Code

### Demo

In this demo, we developed a total of eight algorithms. Four of them are independent algorithm models and the other four are integrated models:
  1. The LR binary model:
  ```bash
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 7 -C 0.001 -gt LME_Al_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 1 -C 0.0001 -gt LME_Co_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 7 -C 0.01 -gt LME_Ni_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 5 -C 0.01 -gt LME_Ti_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 10 -C 0.01 -gt LME_Le_Close 
    python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 10 -C 0.001 -gt LME_Zi_Close
   ```
 * The model performance will be found in `test_result/` folder.
 2.  The LR three classification model:
 ```bash
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 5 -C 0.0001 -gt LME_Al_Close 
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 1 -C 0.0001 -gt LME_Co_Close 
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 10 -C 0.001 -gt LME_Ni_Close
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 1 -C 0.0001 -gt LME_Ti_Close
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 7 -C 0.001 -gt LME_Le_Close 
  python code/train/train_log_reg_online_classifier.py -c exp/online_metal.conf -v v5_ex2 -s 1 -l 7 -C 0.001 -gt LME_Zi_Close
 ```
 3. The XGBoost model:
 ```bash
 python code/train/train_xgboost_online.py -c exp/online_metal.conf -v v5 -s 1 -l 5 -max_depth 3 -learning_rate 0.8 -gamma 0.7 -min_child 3 -subsample 0.7 -gt LME_Al_Close
python code/train/train_xgboost_online.py -c exp/online_metal.conf -v v5 -s 1 -l 3 -max_depth 5 -learning_rate 0.8 -gamma 0.6 -min_child 3 -subsample 0.9 -gt LME_Co_Close
python code/train/train_xgboost_online.py -c exp/online_metal.conf -v v5 -s 1 -l 5 -max_depth 5 -learning_rate 0.8 -gamma 0.9 -min_child 4 -subsample 0.6 -gt LME_Ni_Close
python code/train/train_xgboost_online.py -c exp/online_metal.conf -v v5 -s 1 -l 10 -max_depth 5 -learning_rate 0.9 -gamma 0.6 -min_child 3 -subsample 0.9 -gt LME_Ti_Close
python code/train/train_xgboost_online.py -c exp/online_metal.conf -v v5 -s 1 -l 10 -max_depth 4 -learning_rate 0.6 -gamma 0.7 -min_child 3 -subsample 0.85 -gt LME_Le_Close
python code/train/train_xgboost_online.py -c exp/online_metal.conf -v v5 -s 1 -l 10 -max_depth 5 -learning_rate 0.8 -gamma 0.7 -min_child 3 -subsample 0.6 -gt LME_Zi_Close
 ```
 4. The DA-LSTM model:
 ```bash
  python code/train/train_da_lstm_online.py -c exp/online_metal.conf -v v5 -s 1 -l 3 -hidden 40 -gt LME
 ``` 
 5. The voting model:
 ```bash
  python ijcai_weight_multi.py
 ```
 6. The weight model:
 ```bash
  python weight_multi.py
 ```
 7. The LR weight model:
 ```bash
  python lr_ensemble.py 
 ```
 8. The LD-Ensemble:
 ```bash
  python LD-ensemble.py
 ```



