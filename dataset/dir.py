import pandas as pd
import numpy as np
import torch
import ast
import os

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import random


class RadarDataset(Dataset):
    def __init__(self, data_txt=None, file_path_prefix=None, dir_list=None, max_points=100, max_frames=50, transform=None, method='mask', augment=False):
        """
        Args:
            data_txt: Path to text file listing data files 一个txt文件，里面每一行是一个csv文件的路径
            file_path_prefix: Prefix of the file path
            dir_list: List of directories to read files from 一个列表，里面每一个元素是一个文件夹的名字(目录需要加上前缀file_path_prefix)，这个文件夹里面有很多csv文件
            max_points: Maximum number of points per frame
            max_frames: Maximum number of frames per sample
            transform: Optional transform to apply
            method: Method to handle variable frame numbers ('pad', 'clip', 'random_clip', 'mask')
            augment: Whether to apply data augmentation
        """
        self.max_points = max_points
        self.max_frames = max_frames
        self.transform = transform
        self.method = method
        self.augment = augment
        self.samples = []
        self.files = []

        # 将每一个csv文件的路径加入到self.files中
        if data_txt:
            self.files = [line.strip() for line in open(data_txt, 'r')]
        elif file_path_prefix and dir_list:
            for directory in dir_list:
                full_dir_path = os.path.join(file_path_prefix, directory)
                for root, _, files in os.walk(full_dir_path):
                    for file in files:
                        if file.endswith('.csv'):
                            self.files.append(os.path.join(root, file))

        # Define action to label mapping
        self.action_mapping = {
            # Positive samples
            'fanshen': 0,
            'lying': 0,
            'sit': 0,
            'leave': 0,
            'goBed': 0,
            # Negative samples
            'roll': 1,
            'fallSit': 1,
            'slowFall': 1
        }

        # Count class distribution
        self.class_counts = {0: 0, 1: 0}
        for file in self.files:
            action = file.split('_')[-2]
            label = self.action_mapping.get(action, -1)
            if label in self.class_counts:
                self.class_counts[label] += 1

        print(f"Dataset loaded. Class distribution: {self.class_counts}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        point_clouds, mask, label, frame_num = self.get_data_from_csv(filename)
        point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        frame_num = torch.tensor(frame_num, dtype=torch.int32)
        return point_clouds, mask, label, frame_num


    def apply_augmentation(self, point_cloud):
        """Apply data augmentation to point cloud"""
        # Only augment with 50% probability
        if not self.augment or random.random() < 0.5:
            return point_cloud

        # Clone to avoid modifying original
        augmented = point_cloud.copy()

        # Random noise addition (small Gaussian noise)
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented += noise

        # Random point dropout (randomly zero out some points)
        if random.random() < 0.3:
            mask = np.random.rand(*augmented.shape[:2]) > 0.1
            mask = np.expand_dims(mask, -1).repeat(5, axis=2)
            augmented = augmented * mask

        # Small random rotation around z-axis (for x,y coordinates)
        if random.random() < 0.4:
            angle = np.random.uniform(-0.1, 0.1)  # Small angle in radians
            cos_angle, sin_angle = np.cos(angle), np.sin(angle)
            for i in range(len(augmented)):
                x, y = augmented[i, :, 0], augmented[i, :, 1]
                augmented[i, :, 0] = x * cos_angle - y * sin_angle
                augmented[i, :, 1] = x * sin_angle + y * cos_angle

        return augmented

    # 对每一帧数据进行归一化处理，point_cloud: (point_num, 5)
    def process_pointcloud(self, point_cloud):
        """使用 MinMaxScaler 归一化"""
        # 如果 point_cloud 为空，直接返回全零数组
        if point_cloud.shape[0] == 0:
            point_cloud = np.zeros((self.max_points, 5))
            return point_cloud
        scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化到 [0,1]
        point_cloud = scaler.fit_transform(point_cloud)

        # Pad/truncate
        if len(point_cloud) < self.max_points:
            pad = np.zeros((self.max_points - len(point_cloud), 5))
            point_cloud = np.concatenate([point_cloud, pad])
        else:
            point_cloud = point_cloud[:self.max_points]

        return point_cloud

    def handle_variable_frames(self, point_clouds):
        """Handle variable number of frames using specified method"""
        frame_num = len(point_clouds)

        # 多余的帧数删除，不足的填充0，不需要mask
        if self.method == 'pad':
            if frame_num < self.max_frames:
                pad_frames = np.zeros((self.max_frames - frame_num, self.max_points, 5))
                point_clouds = np.concatenate([point_clouds, pad_frames])
            else:
                point_clouds = point_clouds[:self.max_frames]

            # No mask needed for simple padding
            mask = np.ones(self.max_frames)
            mask[frame_num:] = 0  # Mark padding as 0

        elif self.method == 'clip':
            # Simple clipping: take first max_frames frames
            if frame_num > self.max_frames:
                point_clouds = point_clouds[:self.max_frames]
            else:
                # Still need to pad if fewer frames
                pad_frames = np.zeros((self.max_frames - frame_num, self.max_points, 5))
                point_clouds = np.concatenate([point_clouds, pad_frames])

            # Create mask
            mask = np.ones(self.max_frames)
            mask[frame_num:] = 0

        elif self.method == 'random_clip':
            # Random clipping: randomly select frames if too many
            if frame_num > self.max_frames:
                # Randomly select max_frames indices
                indices = sorted(random.sample(range(frame_num), self.max_frames))
                point_clouds = point_clouds[indices]
            else:
                # Still need to pad if fewer frames
                pad_frames = np.zeros((self.max_frames - frame_num, self.max_points, 5))
                point_clouds = np.concatenate([point_clouds, pad_frames])

            # Create mask
            mask = np.ones(self.max_frames)
            mask[min(frame_num, self.max_frames):] = 0

        elif self.method == 'mask':
            # Mask method: always pad to max_frames and create explicit mask
            if frame_num < self.max_frames:
                pad_frames = np.zeros((self.max_frames - frame_num, self.max_points, 5))
                point_clouds = np.concatenate([point_clouds, pad_frames])
            else:
                point_clouds = point_clouds[:self.max_frames]

            # Create explicit mask (1 for real data, 0 for padding)
            mask = np.ones(self.max_frames)
            mask[frame_num:] = 0

        return point_clouds, mask, frame_num



    def get_data_from_csv(self, filename):
        if filename.endswith('.csv'):
            point_clouds = []  # List to store multiple point cloud samples
            action = filename.split('_')[-2]  # Get action type
            label = self.action_mapping.get(action, -1)  # Avoid KeyError

            df = pd.read_csv(filename)

            for _, row in df.iterrows():
                # Parse point cloud data format(point_num, 5)
                try:
                    point_cloud = np.array(ast.literal_eval(row[1]), dtype=np.float32)
                except (SyntaxError, ValueError):
                    print(f"Warning: {filename} contains incorrect point cloud format, skipping this row!")
                    continue  # Skip erroneous data

                # Ensure point_cloud shape is always (n, 5)
                if point_cloud.ndim == 1:  # Handle empty arrays or incorrect format
                    point_cloud = np.zeros((0, 5), dtype=np.float32)  # Set to empty (0, 5)
                elif point_cloud.shape[1] != 5:  # Ensure column count is 5
                    print(f"Warning: {filename} point cloud column count mismatch, skipping this row!")
                    continue

                # Preprocess (normalize + pad/truncate)
                processed_cloud = self.process_pointcloud(point_cloud)

                # Add to sample list
                point_clouds.append(processed_cloud)

            # Handle variable frame numbers
            if len(point_clouds) == 0:
                # Handle empty case
                point_clouds = np.zeros((self.max_frames, self.max_points, 5))
                mask = np.zeros(self.max_frames)  # All masked
                frame_num = 0
            else:
                # point_clouds shape is (frame_num, point_num, 5)
                point_clouds = np.array(point_clouds)

                # Apply augmentation if enabled
                if self.augment and label == 1:  # Apply augmentation more to minority class
                    point_clouds = self.apply_augmentation(point_clouds)

                point_clouds, mask, frame_num = self.handle_variable_frames(point_clouds)

        return point_clouds, mask, label, frame_num


# point_clouds: [frame_num, point_num, 5]
# mask: (frame_num, )
# label: 0 or 1