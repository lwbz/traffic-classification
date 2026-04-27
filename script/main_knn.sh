#!/usr/bin/env bash

#python ../main_knn.py --is_pseudo 1 --pseudo_epochs 1 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.01 --log_path ../train_log
python ../main_knn.py --is_pseudo 1 --pseudo_epochs 50 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.01 --log_path ../train_log
python ../main_knn.py --is_pseudo 1 --pseudo_epochs 50 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.05 --log_path ../train_log
python ../main_knn.py --is_pseudo 1 --pseudo_epochs 50 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.1 --log_path ../train_log

python ../main_knn.py --sup_con_mu 0.0 --is_pseudo 1 --pseudo_epochs 50 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log


python ../main_knn.py --knn_num_tem 10 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --knn_num_tem 20 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --knn_num_tem 30 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --knn_num_tem 40 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --knn_num_tem 50 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --knn_num_tem 60 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --knn_num_tem 70 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --knn_num_tem 80 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --knn_num_tem 90 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log

#python ../main_knn.py --pseudo_weight 0.1 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --pseudo_weight 0.3 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --pseudo_weight 0.5 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --pseudo_weight 0.7 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --pseudo_weight 0.9 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#
python ../main_knn.py --sigma 0.1 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sigma 0.3 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sigma 0.5 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sigma 0.7 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sigma 0.9 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#
#python ../main_knn.py --alpha 0.1 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --alpha 0.3 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --alpha 0.5 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --alpha 0.7 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
#python ../main_knn.py --alpha 0.9 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log

python ../main_knn.py --sup_con_mu 0.001 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sup_con_mu 0.005 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sup_con_mu 0.01 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sup_con_mu 0.05 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sup_con_mu 0.1 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --sup_con_mu 0.5 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log


python ../main_knn.py --temperature 0.001 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --temperature 0.01 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --temperature 0.1 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --temperature 10 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log
python ../main_knn.py --temperature 30 --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT17 --labeled_ratio 0.03 --log_path ../train_log




