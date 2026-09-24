# dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from mmaction.registry import DATASETS  # 导入 DATASETS 注册表
import cv2

@DATASETS.register_module() 
class MSKEDataset(Dataset):
    def __init__(self, processed_dir, num_segments=4, frames_per_segment=6, frame_size=(256, 256)):
        self.processed_dir = processed_dir
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.frame_size = frame_size
        self.video_dirs = [os.path.join(processed_dir, d) for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]

    def _load_rgb_frames(self, video_dir):
        frames = []
        for img_name in sorted(os.listdir(video_dir)):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(video_dir, img_name)
                frame = cv2.imread(img_path)
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
        return frames

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        frames = self._load_rgb_frames(video_dir)
        segments = []
        step = len(frames) // self.num_segments

        for i in range(self.num_segments):
            start = max(0, i * step - 1)
            end = min(len(frames), i * step + self.frames_per_segment - 1)
            segment = frames[start:end]
            
            if len(segment) < self.frames_per_segment:
                segment += [segment[-1]] * (self.frames_per_segment - len(segment))
            
            segments.append(segment)

        segments = [torch.from_numpy(np.array(seg)).permute(0, 3, 1, 2).float() / 255.0 for seg in segments]
        return segments, video_dir  # Return segments and video directory for saving
