import cv2
import numpy as np
import os
from tqdm import tqdm
from mmagic.models.editors.basicvsr.basicvsr_net import SPyNet
import torch

# Initialize SPyNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spynet = SPyNet(pretrained=None).to(device)
spynet.eval()

def calculate_flow_intensity(frame1, frame2):
    """Calculate the intensity of the optical flow between two frames using SPyNet."""
    # Preprocess frames for SPyNet input
    frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Calculate optical flow
    with torch.no_grad():
        flow = spynet(frame1_tensor, frame2_tensor)

    # Calculate intensity as the magnitude of the optical flow
    flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2).mean().item()
    return flow_magnitude

def extract_key_frames(rgb_video_path, num_key_frames):
    """Extract key frames from the RGB video based on flow intensity changes using SPyNet."""
    cap = cv2.VideoCapture(rgb_video_path)
    flow_intensities = []
    frames = []

    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return []

    frames.append(frame1)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        flow_intensity = calculate_flow_intensity(frame1, frame2)
        flow_intensities.append(flow_intensity)
        frames.append(frame2)
        frame1 = frame2

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
    seen_indices = set()

    for idx in key_frame_indices:
        for offset in range(-frames_per_key // 2 - 1, frames_per_key // 2 + 1):
            new_idx = idx + offset
            if 0 <= new_idx < len(rgb_frames) and new_idx not in seen_indices:
                selected_frames.append(rgb_frames[new_idx])
                seen_indices.add(new_idx)

    if len(selected_frames) > num_total_frames + len(key_frame_indices) * 2:
        selected_frames = selected_frames[:num_total_frames + len(key_frame_indices) * 2]
    elif len(selected_frames) < num_total_frames + len(key_frame_indices) * 2:
        selected_frames.extend([selected_frames[-1]] * (num_total_frames + len(key_frame_indices) * 2 - len(selected_frames)))

    return selected_frames

def save_rgb_frames(frames, output_folder, label):
    os.makedirs(output_folder, exist_ok=True)
    for i, rgb in enumerate(frames):
        cv2.imwrite(os.path.join(output_folder, f'rgb_{i}.jpg'), rgb)

    with open(os.path.join(output_folder, 'label.txt'), 'w') as f_label:
        f_label.write(f"{label}\n")

def preprocess_videos(rgb_root, ann_file_rgb, output_dir, num_key_frames=4, num_total_frames=16):
    os.makedirs(output_dir, exist_ok=True)

    with open(ann_file_rgb) as f_rgb:
        rgb_lines = f_rgb.readlines()

    for line_rgb in tqdm(rgb_lines, total=len(rgb_lines)):
        filename_rgb = os.path.join(rgb_root, line_rgb.strip().split()[0])
        label = line_rgb.strip().split()[1]

        key_frame_indices = extract_key_frames(filename_rgb, num_key_frames)
        selected_rgb_frames = extract_rgb_frames(filename_rgb, key_frame_indices, num_total_frames)

        video_name = os.path.splitext(os.path.basename(filename_rgb))[0]
        video_output_folder = os.path.join(output_dir, video_name)
        save_rgb_frames(selected_rgb_frames, video_output_folder, label)

if __name__ == "__main__":
    rgb_root = '/root/mmaction2/data/mydatasets_video/RGB_video'
    ann_file_rgb = '/root/mmaction2/data/mydatasets_video/RGB_txt/testlist.txt'
    output_dir = '/root/mmaction2/projects/my_project/datasets/test'

    preprocess_videos(rgb_root, ann_file_rgb, output_dir)
