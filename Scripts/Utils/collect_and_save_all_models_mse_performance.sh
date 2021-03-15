#!/bin/bash

cd $HOME/Documents/Working/COVID19TrendPrediction

n_in="1 5 7 14 30"
n_out="1 5 7 14 30"


for var1 in ${n_in[@]}; do
    for var2  in ${n_out[@]}; do
        # echo "python3 Models/xgboost_model.py $var1 $var2"
        # python3 Models/previous_day.py $var1 $var2
        # echo "python3 Scripts/Utils/collect_and_save_all_models_mse_performance.py $var1 $var2 --aggr --save"
        # python3 Scripts/Utils/collect_and_save_all_models_mse_performance.py $var1 $var2 --aggr --save
        python3 Scripts/Utils/collect_and_save_all_models_mse_performance.py $var1 $var2 --aggr --save --model_param_epoch 100
    done
done
