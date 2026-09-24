import cv2
import numpy as np
import os
from tqdm import tqdm

def calculate_flow_intensity(flow_frame):
    """Calculate the intensity of the optical flow frame by combining the magnitude (value), direction (hue), and saturation (saturation)."""
    hsv_frame = cv2.cvtColor(flow_frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_frame)
    hue = hue / 180.0
    saturation = saturation / 255.0
    value = value / 255.0
    intensity = np.sum(hue * saturation * value)
    return intensity

def extract_key_frames(flow_video_path, num_key_frames):
    """Extract key frames from the optical flow video based on flow intensity changes."""
    cap = cv2.VideoCapture(flow_video_path)
    flow_intensities = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        flow_intensity = calculate_flow_intensity(frame)
        flow_intensities.append(flow_intensity)

    cap.release()
    intensity_changes = np.diff(flow_intensities)
    intensity_changes = np.abs(intensity_changes)
    key_frame_indices = np.argsort(intensity_changes)[-num_key_frames:]
    key_frame_indices = np.sort(key_frame_indices)

    return key_frame_indices

def extract_rgb_frames(rgb_video_path, key_frame_indices, num_total_frames):
    cap_rgb = cv2.VideoCapture(rgb_video_path)

    rgb_frames = []

    while True:
        ret_rgb, frame_rgb = cap_rgb.read()
        if not ret_rgb:
            break
        rgb_frames.append(frame_rgb)

    cap_rgb.release()

    selected_frames = []
    frames_per_key = num_total_frames // len(key_frame_indices)
    seen_indices = set()  # 用于避免重复提取帧

    # 修改后的逻辑：每段保留前后各1帧，但避免重复提取帧
    for idx in key_frame_indices:
        for offset in range(-frames_per_key // 2 - 1, frames_per_key // 2 + 1):
            new_idx = idx + offset
            if 0 <= new_idx < len(rgb_frames) and new_idx not in seen_indices:
                selected_frames.append(rgb_frames[new_idx])
                seen_indices.add(new_idx)

    # 检查帧数是否满足要求，尽量避免截断
    if len(selected_frames) > num_total_frames + len(key_frame_indices) * 2:
        selected_frames = selected_frames[:num_total_frames + len(key_frame_indices) * 2]
    elif len(selected_frames) < num_total_frames + len(key_frame_indices) * 2:
        selected_frames.extend([selected_frames[-1]] * (num_total_frames + len(key_frame_indices) * 2 - len(selected_frames)))

    return selected_frames


def save_rgb_frames(frames, output_folder, label):
    os.makedirs(output_folder, exist_ok=True)
    for i, rgb in enumerate(frames):
        cv2.imwrite(os.path.join(output_folder, f'rgb_{i}.jpg'), rgb)

    # 保存标签
    with open(os.path.join(output_folder, 'label.txt'), 'w') as f_label:
        f_label.write(f"{label}\n")

def preprocess_videos(rgb_root, flow_root, ann_file_rgb, ann_file_flow, output_dir, num_key_frames=4, num_total_frames=16):
    os.makedirs(output_dir, exist_ok=True)

    with open(ann_file_rgb) as f_rgb, open(ann_file_flow) as f_flow:
        rgb_lines = f_rgb.readlines()
        flow_lines = f_flow.readlines()

    if len(rgb_lines) != len(flow_lines):
        raise ValueError("RGB and Flow annotation files must have the same number of lines")

    for line_rgb, line_flow in tqdm(zip(rgb_lines, flow_lines), total=len(rgb_lines)):
        filename_rgb = os.path.join(rgb_root, line_rgb.strip().split()[0])
        filename_flow = os.path.join(flow_root, line_flow.strip().split()[0])
        label = line_rgb.strip().split()[1]

        # 提取关键帧索引
        key_frame_indices = extract_key_frames(filename_flow, num_key_frames)
        
        # 基于关键帧索引提取RGB帧
        selected_rgb_frames = extract_rgb_frames(filename_rgb, key_frame_indices, num_total_frames)

        # 保存RGB帧到输出目录
        video_name = os.path.splitext(os.path.basename(filename_rgb))[0]
        video_output_folder = os.path.join(output_dir, video_name)
        save_rgb_frames(selected_rgb_frames, video_output_folder, label)

if __name__ == "__main__":
    # RGB 视频根目录
    rgb_root = '/root/mmaction2/data/mydatasets_video/RGB_video'

    # Flow 视频根目录
    flow_root = '/root/mmaction2/data/mydatasets_video/flow_video'

    # 标注文件路径
    ann_file_rgb = '/root/mmaction2/data/mydatasets_video/RGB_txt/vallist.txt'
    ann_file_flow = '/root/mmaction2/data/mydatasets_video/flow_txt/vallist.txt'

    # 输出目录
    output_dir = '/root/mmaction2/data/mydatasets_video/rawframes_16/val'

    preprocess_videos(rgb_root, flow_root, ann_file_rgb, ann_file_flow, output_dir)
