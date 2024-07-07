import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
df_sample = pd.read_csv('data/ag_news_train_clean_sample.csv')
# ramdom seed
np.random.seed(42)
def replace_and_visualize_confusion_matrix(data, target_class, replacement_class, num_replacements):
    # 找出目標類別的索引
    target_indices = np.where(data == target_class)[0]
    
    # 隨機選擇指定數量的目標類別索引
    indices_to_replace = np.random.choice(target_indices, num_replacements, replace=False)
    
    # 保存原始數據作為實際標籤
    actual_labels = data.copy()
    
    # 將選中的索引對應的值替換為替代類別
    data[indices_to_replace] = replacement_class
    
    # 將替換後的數據作為預測標籤
    predicted_labels = data
    
    # 計算混淆矩陣
    cm = confusion_matrix(actual_labels, predicted_labels)

    return predicted_labels,actual_labels,cm
def plot_multiple_confusion_matrices(confusion_matrices, titles,target,replace):
    num_matrices = len(confusion_matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(20, 5))

    for i, ax in enumerate(axes):
        sns.heatmap(confusion_matrices[i], annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=np.arange(4), yticklabels=np.arange(4))
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(f'confusion_matrices{target}_{replace}.png')
    
def gen_ccn (data,initial_class):
    data['label'] = initial_class
    return data
# 示例數據：假設這是你的 AG News 數據集的標籤
df_sample_i = df_sample.copy()
data1 = df_sample_i['label'].values
df_sample_i = df_sample.copy()
data2 = df_sample_i['label'].values
df_sample_i = df_sample.copy()
data3 =  df_sample_i['label'].values
df_sample_i = df_sample.copy()
data4 =  df_sample_i['label'].values


# 使用函數替換類別並生成混淆矩陣
target = 2
replacement = 0
# 0 
preic_0 ,actual_0,cm_0=replace_and_visualize_confusion_matrix(data1, target_class=target, replacement_class=replacement, num_replacements=0)
#10,20,30,40
preic_10,actual_10,cm_10=replace_and_visualize_confusion_matrix(data1, target_class=target, replacement_class=replacement, num_replacements=10)

preic_20,actual_20,cm_20=replace_and_visualize_confusion_matrix(data2, target_class=target, replacement_class=replacement, num_replacements=20)

preic_30,actual_30,cm_30=replace_and_visualize_confusion_matrix(data3, target_class=target, replacement_class=replacement, num_replacements=30)

preic_40,actual_40,cm_40=replace_and_visualize_confusion_matrix(data4, target_class=target, replacement_class=replacement, num_replacements=40)
print(preic_0,actual_0)
# plot_multiple_confusion_matrices([cm_0,cm_10, cm_20, cm_30, cm_40], ['0_Replacements','10 Replacements', '20 Replacements', '30 Replacements', '40 Replacements'],target,replacement)

ccn_0 = gen_ccn(df_sample,preic_0)
ccn_0 = df_sample [['label','text']]
ccn_10 = gen_ccn(df_sample,preic_10)
ccn_10 = ccn_10 [['label','text']]
ccn_20 = gen_ccn(df_sample,preic_20)
ccn_20 = ccn_20 [['label','text']]
ccn_30 = gen_ccn(df_sample,preic_30)
ccn_30 = ccn_30 [['label','text']]
ccn_40 = gen_ccn(df_sample,preic_40)
ccn_40 = ccn_40 [['label','text']]
if os.path.exists(f'data/agnews{target}') == False:
    os.makedirs(f'data/agnews{target}')

ccn_0.to_csv(f'data/agnews{target}/ag_news_ccn_sample0_label:t{target}to{replacement}.csv', index=False,header=False)   
ccn_10.to_csv(f'data/agnews{target}/ag_news_ccn_sample10_label:t{target}to{replacement}.csv', index=False,header=False)
ccn_20.to_csv(f'data/agnews{target}/ag_news_ccn_sample20_label:t{target}to{replacement}.csv', index=False,header=False)
ccn_30.to_csv(f'data/agnews{target}/ag_news_ccn_sample30_label:t{target}to{replacement}.csv', index=False,header=False)
ccn_40.to_csv(f'data/agnews{target}/ag_news_ccn_sample40_label:t{target}to{replacement}.csv', index=False,header=False)

