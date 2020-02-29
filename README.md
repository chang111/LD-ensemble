
## Dependencies

* python 3.6
* [PyTorch 0.4.0](https://pytorch.org/get-started/locally/)
* [XGBoost 0.8](https://pypi.org/project/xgboost/)

## The Framework structure
```directory
LD-ensemble/
      code/
        data/
              load_data.py
        model/
            base_predictor.py
            DALSTM.py
            logistic_regression.py
        train/
            train_da_lstm.py
            train_da_lstm_online.py
            train_log_reg.py
            train_log_reg_online.py
            train_log_reg_online_classifier.py
            train_log_reg_three_classifier.py
            train_xgboost.py
            train_xgboost_online.py
        utils/
            construct_data.py
            evaludator.py
            log_reg_functions.py
            normalize_feature.py
            process_strategy.py
            read_data.py
            readme.md
            Technical_indicator.py
            transform_data.py
            version_control_functions.py
        LD-ensemble.py
        lr_ensemble.py
        weight_multi.py
      data/
          DALSTM_test_result/
          Label/
          LME_2019/
          lr_test_result/
          lr_three_result/
          xgboost_test_result/
      exp/
          online_metal.conf
          strat_param_v10.conf
          strat_param_v11.conf
          strat_param_v12.conf
          strat_param_v14.conf
          strat_param_v18.conf
          strat_param_v20.conf
          strat_param_v9.conf
      README.md
```
## Code

### Demo

In this demo, we developed a total of eight algorithms. Four of them are single algorithm models and the other four are integrated models:
  * At first, we will train the model. And the parameter for four single algorithm is:
    | Model | parameter |
| :-----: | :-----: |
| LR-binary | lag:[1,3,5,7,10] |
| LR-three-class | lag:[1,3,5,7,10] |
| XGB | lag:[1,3,5,7,10] |
| DA-RNN | lag:[1,3,5,7,10], hidden:[10,20,30,40,50] |
  * Then we will choose the best parameter from the validation result.
  * At the last we run the model online file with the best parameter. For example:
  ```bash
  >>python code/train/train_log_reg_online.py -c exp/online_metal.conf -v v5 -s 1 -l 7 -C 0.001 -gt LME_Al_Close
  ``` 
  * After the test finishes, we can get the prediction result under the folder `data`.

### Test the result
  We have put the three code file `LD-ensemble.py` `lr_ensemble.py` `weight_multi.py`
  * You can get the three single model result, voting result and weight result by runing:
  ```bash
  >>python code/weight_multi.py
  ``` 
  * You can get the stacking result by running:
  ```bash
  >>python code/lr_ensemble.py
  ```   
  * You can get the LD-stacking result by running:
  ```bash
  >>python code/LD-ensemble.py
  ```    



