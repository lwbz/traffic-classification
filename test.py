import pickle
import dgl

# dataset_info = pickle.load(open(r'D:\code\python\practice\flow\tmp\Medboit_info_20.pickle', 'rb'))

# data = pickle.load(open(r'D:\code\python\practice\flow\tmp\Medboit_all_seq_20.pickle', 'rb'))
# data = pickle.load(open(r'D:\code\python\practice\flow\tmp\IoT_all_seq_30.pickle', 'rb'))
# data1 = pickle.load(open(r'./Medboit_all_labels_20.pickle', 'rb'))
# data2 = pickle.load(open(r'./Medboit_all_seq_20.pickle', 'rb'))

data1 = pickle.load(open(r'./ustctfc_20_train_graph.pickle', 'rb'))
data2 = pickle.load(open(r'./ustctfc_20_test_graph.pickle', 'rb'))
data3 = pickle.load(open(r'./ustctfc_20_valid_graph.pickle', 'rb'))

# print(dataset_info)
# print(data)
print(len(data1))
print(len(data2))
print(len(data3))
#
#
#
#
# import pickle
# import pandas as pd
# import numpy as np
#
# # 读取pickle文件
# with open('Medboit_all_labels_20.pickle', 'rb') as f:
#     labels = pickle.load(f)
#
# with open('Medboit_all_seq_20.pickle', 'rb') as f:
#     sequences = pickle.load(f)
#
# # 确保数据长度一致
# assert len(labels) == len(sequences) == 8349, "数据长度不匹配"
#
# # 合并相邻同类数据
# combined_data = []
# combined_labels = []
# current_label = labels[0]
# current_seq = list(sequences[0])
#
# for i in range(1, len(labels)):
#     if labels[i] == current_label:
#         # 同类，继续合并
#         current_seq.extend(sequences[i])
#     else:
#         # 不同类，保存当前合并结果
#         combined_data.append(current_seq)
#         combined_labels.append(current_label)
#         # 重置为新类
#         current_label = labels[i]
#         current_seq = list(sequences[i])
#
# # 保存最后一组
# combined_data.append(current_seq)
# combined_labels.append(current_label)
#
# # 分割成50个数据包的序列
# final_data = []
# final_labels = []
#
# for label, seq in zip(combined_labels, combined_data):
#     # 按50个数据包分割
#     for i in range(0, len(seq), 50):
#         if i + 50 <= len(seq):  # 确保有足够的50个数据
#             final_data.append(seq[i:i+50])
#             final_labels.append(label)
#
# # 创建DataFrame
# columns = ['label'] + [f'packet_{i+1}' for i in range(50)]
# df = pd.DataFrame(columns=columns)
#
# # 填充数据
# df['label'] = final_labels
# for i in range(50):
#     df[f'packet_{i+1}'] = [row[i] for row in final_data]
#
# # 保存为CSV
# df.to_csv('medboit_combined.csv', index=False)
#
# print("CSV文件已生成：medboit_combined.csv")
#
#

# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('medboit_combined.csv')
#
# # 将标签列的值加1
# df['label'] = df['label'] + 1
#
# # 保存修改后的CSV文件
# df.to_csv('medboit_combined.csv', index=False)
#
# print("标签值已加1，新的CSV文件已生成：medboit_combined.csv")

#
# import pickle
# import pandas as pd
#
# # 读取pickle文件
# with open('ustctfc_all_labels_20.pickle', 'rb') as f:
#     labels = pickle.load(f)
#
# with open('ustctfc_all_seq_20.pickle', 'rb') as f:
#     sequences = pickle.load(f)
#
# # 确保数据长度一致
# # assert len(labels) == len(sequences) == 8349, "数据长度不匹配"
#
# # 创建DataFrame
# columns = ['label'] + [f'packet_{i+1}' for i in range(20)]
# df = pd.DataFrame(columns=columns)
#
# # 填充数据
# df['label'] = labels
# for i in range(20):
#     df[f'packet_{i+1}'] = [seq[i] for seq in sequences]
#
# # 保存为CSV
# df.to_csv('ustctfc.csv', index=False)
#
# ndf = pd.read_csv('ustctfc.csv')
#
# # 将标签列的值加1
# ndf['label'] = ndf['label'] + 1
#
# # 保存修改后的CSV文件
# ndf.to_csv('ustctfc.csv', index=False)
#
# print("CSV文件已生成：ustctfc.csv")

