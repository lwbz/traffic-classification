#!/usr/bin/env bash

#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.01 --knn_num_tem 40 --log_path ../train_log
#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.01 --knn_num_tem 40 --log_path ../train_log
#
#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.03 --knn_num_tem 50 --log_path ../train_log
#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.03 --knn_num_tem 50 --log_path ../train_log
#
#
#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.05 --knn_num_tem 50 --log_path ../train_log
#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.05 --knn_num_tem 50 --log_path ../train_log
#
#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.1 --knn_num_tem 50 --log_path ../train_log
#python ../main_time_only.py --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset IOT21 --labeled_ratio 0.1 --knn_num_tem 50 --log_path ../train_log

python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.01 --log_path ../train_log
python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.01 --log_path ../train_log

python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.03 --log_path ../train_log
python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.03 --log_path ../train_log

python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.05 --log_path ../train_log
python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.05 --log_path ../train_log

python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.10 --log_path ../train_log
python ../main_time_only.py --warmup_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.10 --log_path ../train_log



python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.01 --knn_num_tem 50 --log_path ../train_log
python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.01 --knn_num_tem 50 --log_path ../train_log

python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.03 --knn_num_tem 50 --log_path ../train_log
python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.03 --knn_num_tem 50 --log_path ../train_log

python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.05 --knn_num_tem 50 --log_path ../train_log
python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.05 --knn_num_tem 50 --log_path ../train_log

python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.1 --knn_num_tem 50 --log_path ../train_log
python ../main_time_only.py --is_pseudo 1 --pseudo_epochs 30 --dataroot ../IOT_Dataset --dataset TMC13 --labeled_ratio 0.1 --knn_num_tem 50 --log_path ../train_log





