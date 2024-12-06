import os
from PIL import Image
import numpy as np
import tifffile as tiff


def stack_channels_and_save(source_dir, save_dir):
    # 通道名称（根据您实际的图像文件名调整）
    channels = ['red', 'green', 'blue', 'yellow']

    # 获取目录中的所有文件
    files = os.listdir(source_dir)

    # 根据文件名前缀对文件进行分组
    grouped_files = {}

    # 遍历文件并按前缀分组
    for file in files:
        if file.endswith('.tif'):
            # 获取图像的前缀，假设后缀是"_red.tif", "_green.tif", etc.
            prefix = "_".join(file.split('_')[:-1])
            if prefix not in grouped_files:
                grouped_files[prefix] = {}
            # 根据通道将文件分类
            for channel in channels:
                if channel in file:
                    grouped_files[prefix][channel] = os.path.join(source_dir, file)
                    break

    # 对每个前缀，堆叠四个通道并保存为新的TIFF文件
    for prefix, channel_files in grouped_files.items():
        if all(channel in channel_files for channel in channels):
            # 读取每个通道的图像
            red_img = Image.open(channel_files['red'])
            green_img = Image.open(channel_files['green'])
            blue_img = Image.open(channel_files['blue'])
            yellow_img = Image.open(channel_files['yellow'])

            # 将图像转换为numpy数组
            red_array = np.array(red_img)
            green_array = np.array(green_img)
            blue_array = np.array(blue_img)
            yellow_array = np.array(yellow_img)

            # 堆叠图像通道
            stacked_image = np.stack([red_array, green_array, blue_array, yellow_array], axis=-1)  # Shape: (H, W, 4)

            # 保存为新文件（文件名只有前缀）
            save_path = os.path.join(save_dir, f"{prefix}.tif")  # 这里没有 _stacked 后缀
            tiff.imsave(save_path, stacked_image)
            print(f"Saved stacked image: {save_path}")
        else:
            print(f"Warning: Missing one or more channels for {prefix}, skipping.")


# 使用示例
source_dir = '/g/data/au38/yy3740/hpa_train_full'  # 替换为您实际的图像文件夹路径
save_dir = '/g/data/au38/yy3740/hpa_train_full_stack'  # 替换为您希望保存堆叠图像的目录
os.makedirs(save_dir, exist_ok=True)

# 执行堆叠操作
stack_channels_and_save(source_dir, save_dir)
