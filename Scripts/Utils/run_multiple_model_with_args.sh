#!/bin/bash

cd $HOME/Documents/Working/COVID19TrendPrediction
# export $WANDB_DISABLE_CODE=true

# n_in="1 7 14 30"
# n_in="7"
# n_in="14 30"
n_in="14"
# n_out="1 7 14 30"
# n_out="14 30"
n_out="1"

# n_in="7"
# n_out="1"

# ~/scratches

# ~/run them all 
for var1 in ${n_in[@]}; do
    for var2  in ${n_out[@]}; do

        # python3 Models/previous_day.py $var1 $var2 --is_test_environment --states Oklahoma --experiment 3  --save_wandb 
        # python3 Models/linear_regressoin.py $var1 $var2 --is_test_environment  --states Oklahoma --experiment 3 --save_wandb
        # python3 Models/mlp.py  $var1 $var2 --model_param_epoch 20 --dont_create_new_model_on_each_run --is_test_environment --train_model_with_1_run --evaluate_on_many_test_data_per_run --states Oklahoma --experiment 3 --save_wandb --repeat 1
        python3 Models/conv1d.py  $var1 $var2 --model_param_epoch 100 --dont_create_new_model_on_each_run --is_test_environment --train_model_with_1_run --evaluate_on_many_test_data_per_run --states Oklahoma --experiment 3 --save_wandb --repeat 1
        python3 Models/lstm.py  $var1 $var2 --model_param_epoch 100 --dont_create_new_model_on_each_run --is_test_environment --train_model_with_1_run --evaluate_on_many_test_data_per_run --states Oklahoma --experiment 3 --save_wandb --repeat 1

    done
done


# # ~/run them all 
# for var1 in ${n_in[@]}; do
#     for var2  in ${n_out[@]}; do
#         # echo "python3 models/xgboost_model.py $var1 $var2"
#         python3 models/previous_day.py $var1 $var2 --save_wandb --is_distributed

#     done
# done

# for var1 in ${n_in[@]}; do
#     for var2  in ${n_out[@]}; do
#         # echo "python3 Models/xgboost_model.py $var1 $var2"
#         python3 Models/linear_regressoin.py $var1 $var2 --save_wandb --is_distributed
#         # python3 Models/linear_regressoin.py $var1 $var2 --save_wandb 
#     done
# done

# for var1 in ${n_in[@]}; do
#     for var2  in ${n_out[@]}; do
#         # echo "python3 Models/xgboost_model.py $var1 $var2"
#         # python3 Models/xgboost_model.py $var1 $var2 --save_wandb --is_distributed
#         python3 Models/xgboost_model.py $var1 $var2 --save_wandb 
#     done
# done

# for var1 in ${n_in[@]}; do
#     for var2  in ${n_out[@]}; do
#         python3 Models/conv1d.py $var1 $var2 --model_param_epoch 100 --save_wandb
#     done
# done

# for var1 in ${n_in[@]}; do
#     for var2  in ${n_out[@]}; do
#         # echo "python3 Models/xgboost_model.py $var1 $var2"
#         # python3 Models/mlp.py $var1 $var2 --model_param_epoch 500 --save_wandb --is_distributed
#         python3 Models/mlp.py $var1 $var2 --model_param_epoch 100 --save_wandb # --is_distributed
#     done
# done

# for var1 in ${n_in[@]}; do
#     for var2  in ${n_out[@]}; do
#         # python3 Models/lstm.py $var1 $var2 --model_param_epoch 500 --save_wandb # --is_distributed
#         python3 Models/lstm.py $var1 $var2 --model_param_epoch 100 --save_wandb # --is_distributed
#     done
# done
