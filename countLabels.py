import pandas as pd

# 读取每个 split 的 CSV 文件
split_0 = pd.read_csv('/Users/yanyiqiao/Downloads/CLAM_RA/results/task_3_tgca_clam_sb_k3_lr1e-4/task_3_tgca_clam_sb_s1/splits_0.csv')
split_1 = pd.read_csv('/Users/yanyiqiao/Downloads/CLAM_RA/results/task_3_tgca_clam_sb_k3_lr1e-4/task_3_tgca_clam_sb_s1/splits_1.csv')
split_2 = pd.read_csv('/Users/yanyiqiao/Downloads/CLAM_RA/results/task_3_tgca_clam_sb_k3_lr1e-4/task_3_tgca_clam_sb_s1/splits_2.csv')


train_csv = pd.read_csv('/Users/yanyiqiao/Downloads/CLAM_RA/Labels/TrainLabels1.csv')


def get_label_counts(split, train_csv):
    # 分别提取 train、val 和 test slide_id 列
    train_slide_ids = split['train']
    val_slide_ids = split['val']
    test_slide_ids = split['test']

    # 分别根据 slide_id 匹配训练集中的标签
    train_labels = train_csv[train_csv['slide_id'].isin(train_slide_ids)]
    val_labels = train_csv[train_csv['slide_id'].isin(val_slide_ids)]
    test_labels = train_csv[train_csv['slide_id'].isin(test_slide_ids)]

    # 统计每个 split 中的 label 分布
    train_label_counts = train_labels['label'].value_counts()
    val_label_counts = val_labels['label'].value_counts()
    test_label_counts = test_labels['label'].value_counts()

    return train_label_counts, val_label_counts, test_label_counts


# 分别计算 split_0 的 label 分布
train_0, val_0, test_0 = get_label_counts(split_0, train_csv)
print("Split 0 - Train label distribution:\n", train_0)
print("Split 0 - Validation label distribution:\n", val_0)
print("Split 0 - Test label distribution:\n", test_0)

# 分别计算 split_1 的 label 分布
train_1, val_1, test_1 = get_label_counts(split_1, train_csv)
print("Split 1 - Train label distribution:\n", train_1)
print("Split 1 - Validation label distribution:\n", val_1)
print("Split 1 - Test label distribution:\n", test_1)

# 分别计算 split_2 的 label 分布
train_2, val_2, test_2 = get_label_counts(split_2, train_csv)
print("Split 2 - Train label distribution:\n", train_2)
print("Split 2 - Validation label distribution:\n", val_2)
print("Split 2 - Test label distribution:\n", test_2)