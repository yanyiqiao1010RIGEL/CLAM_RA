import pandas as pd
import numpy as np

# 定义类别总数
NUM_CLASSES = 28

# 读取原始 CSV 文件
input_csv = "train_hpa.csv"
output_csv = "train_hpa_onehot.csv"
df = pd.read_csv(input_csv)

# 定义一个函数，将标签转化为多热编码
def multi_hot_encode(labels, num_classes):
    # 创建全零数组
    one_hot = np.zeros(num_classes, dtype=np.int32)
    # 将标签对应的索引置为 1
    for label in map(int, labels.split(' ')):
        one_hot[label] = 1
    return one_hot

# 对 'label' 列进行多热编码
df['onehot_label'] = df['label'].apply(lambda x: multi_hot_encode(x, NUM_CLASSES))

# 将 onehot_label 转化为字符串（便于保存）
df['onehot_label'] = df['onehot_label'].apply(lambda x: ' '.join(map(str, x)))

# 保存到新 CSV 文件
df[['slide_id', 'onehot_label']].to_csv(output_csv, index=False)

print(f"多热编码完成！结果保存在: {output_csv}")
