#!/usr/bin/env bash

#python ../main_base_graph.py --backbone GCN --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
#python ../main_base_graph.py --backbone GAT --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
#python ../main_base_graph.py --backbone GIN --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
#python ../main_base_graph.py --backbone GraphIoT --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
#python ../main_base.py --backbone MLP --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
#python ../main_base.py --backbone LSTM --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log

python ../main_base_graph.py --backbone DAPP --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
python ../main_base_graph.py --backbone DAPP --dataroot ../IOT_Dataset --labeled_ratio 0.03 --log_path ../train_log
python ../main_base_graph.py --backbone DAPP --dataroot ../IOT_Dataset --labeled_ratio 0.05 --log_path ../train_log
python ../main_base_graph.py --backbone DAPP --dataroot ../IOT_Dataset --labeled_ratio 0.10 --log_path ../train_log


python ../main_base_graph.py --backbone GCN --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
python ../main_base_graph.py --backbone GCN --dataroot ../IOT_Dataset --labeled_ratio 0.03 --log_path ../train_log
python ../main_base_graph.py --backbone GCN --dataroot ../IOT_Dataset --labeled_ratio 0.05 --log_path ../train_log
python ../main_base_graph.py --backbone GCN --dataroot ../IOT_Dataset --labeled_ratio 0.10 --log_path ../train_log

python ../main_base_graph.py --backbone GAT --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
python ../main_base_graph.py --backbone GAT --dataroot ../IOT_Dataset --labeled_ratio 0.03 --log_path ../train_log
python ../main_base_graph.py --backbone GAT --dataroot ../IOT_Dataset --labeled_ratio 0.05 --log_path ../train_log
python ../main_base_graph.py --backbone GAT --dataroot ../IOT_Dataset --labeled_ratio 0.10 --log_path ../train_log

python ../main_base_graph.py --backbone GIN --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
python ../main_base_graph.py --backbone GIN --dataroot ../IOT_Dataset --labeled_ratio 0.03 --log_path ../train_log
python ../main_base_graph.py --backbone GIN --dataroot ../IOT_Dataset --labeled_ratio 0.05 --log_path ../train_log
python ../main_base_graph.py --backbone GIN --dataroot ../IOT_Dataset --labeled_ratio 0.10 --log_path ../train_log

python ../main_base_graph.py --backbone GraphIoT --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
python ../main_base_graph.py --backbone GraphIoT --dataroot ../IOT_Dataset --labeled_ratio 0.03 --log_path ../train_log
python ../main_base_graph.py --backbone GraphIoT --dataroot ../IOT_Dataset --labeled_ratio 0.05 --log_path ../train_log
python ../main_base_graph.py --backbone GraphIoT --dataroot ../IOT_Dataset --labeled_ratio 0.10 --log_path ../train_log

python ../main_base.py --backbone LSTM --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
python ../main_base.py --backbone LSTM --dataroot ../IOT_Dataset --labeled_ratio 0.03 --log_path ../train_log
python ../main_base.py --backbone LSTM --dataroot ../IOT_Dataset --labeled_ratio 0.05 --log_path ../train_log
python ../main_base.py --backbone LSTM --dataroot ../IOT_Dataset --labeled_ratio 0.10 --log_path ../train_log

python ../main_base.py --backbone MLP --dataroot ../IOT_Dataset --labeled_ratio 0.01 --log_path ../train_log
python ../main_base.py --backbone MLP --dataroot ../IOT_Dataset --labeled_ratio 0.03 --log_path ../train_log
python ../main_base.py --backbone MLP --dataroot ../IOT_Dataset --labeled_ratio 0.05 --log_path ../train_log
python ../main_base.py --backbone MLP --dataroot ../IOT_Dataset --labeled_ratio 0.10 --log_path ../train_log




