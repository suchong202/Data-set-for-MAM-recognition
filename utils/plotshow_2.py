import os
import numpy as np
import matplotlib.pyplot as plt

# 指定包含 .npy 文件的文件夹路径
folder_path = '/root/mmaction2/data/mydatasets_video/ROI_test/output/A_nzb_20230206_173642_6'  # 替换为你的文件夹路径
output_folder = '/root/mmaction2/data/mydatasets_video/ROI_test/figure/A_nzb_20230206_173642_6'  # 输出文件夹路径

# 创建输出文件夹（如果不存在的话）
os.makedirs(output_folder, exist_ok=True)

def enhance_contrast(data, low=0.05, high=0.95):
    """拉伸像素值以增强对比度."""
    min_val, max_val = np.percentile(data, (low * 100, high * 100))
    data = np.clip((data - min_val) / (max_val - min_val), 0, 1)
    return data

# 遍历文件夹中的所有 .npy 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 .npy 文件
        data = np.load(file_path)

        # 处理和可视化数据
        if data.ndim == 2:  # 二维图像
            data = enhance_contrast(data)  # 增强对比度
            plt.imshow(data, cmap='gray')
            plt.axis('off')
        elif data.ndim == 3 and data.shape[0] == 3:  # 三维数组，第一维为通道
            data = data.transpose(1, 2, 0)  # 转换为 (height, width, channels)
            data = enhance_contrast(data)  # 增强对比度

            if data.dtype == np.float32 or data.dtype == np.float64:
                data = np.clip(data, 0, 1)
            else:  # 假设是整数类型
                data = np.clip(data, 0, 255)
                data = data.astype(np.uint8)

            plt.imshow(data)
            plt.axis('off')
        else:
            print(f"文件 {filename} 的数据维度不支持可视化")
            continue

        # 保存可视化结果
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
        plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭当前图像以释放内存

print("所有图像已保存到", output_folder)
