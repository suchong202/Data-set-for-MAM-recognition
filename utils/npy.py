import numpy as np

file_path = '/root/mmaction2/data/mydatasets_video/ROI_test/output/A_nzb_20230206_173332_4/roi_weights_segment_3_frame_1.npy'  # 替换为你的 .npy 文件路径
data = np.load(file_path, allow_pickle=True)

print(data)  # 打印数据内容
print("Data shape:", data.shape)  # 打印数据形状
print("Data type:", data.dtype)  # 打印数据类型
print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean())
print("Standard deviation:", data.std())
