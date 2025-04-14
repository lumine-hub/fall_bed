import pandas as pd
import numpy as np
import torch
import ast
from sklearn.preprocessing import MinMaxScaler
from model import EnhancedTCN, LightweightSpatioTemporalModel, EnhancedGRU, HybridModel
import random
# 新添加的目录扫描逻辑
import os
import glob
class Predictor:
    def __init__(self, model_path, model_type='HybridModel', device='auto',
                 max_frames=40, max_points=100, method='mask'):
        # 设备检测
        self.device = torch.device(device if device != 'auto' else
                                   "cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        model_classes = {
            'HybridModel': HybridModel,
            'EnhancedTCN': EnhancedTCN,
            'LightweightSpatioTemporalModel': LightweightSpatioTemporalModel,
            'EnhancedGRU': EnhancedGRU
        }

        self.model = model_classes[model_type](num_classes=2, dropout=0.5).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 数据参数
        self.max_frames = max_frames
        self.max_points = max_points
        self.method = method
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _process_single_csv(self, filename):
        """处理单个CSV文件（与你的RadarDataset保持兼容）"""
        try:
            df = pd.read_csv(filename)
            point_clouds = []

            for _, row in df.iterrows():
                try:
                    point_cloud = np.array(ast.literal_eval(row[1]), dtype=np.float32)
                except (SyntaxError, ValueError):
                    continue  # 跳过错误数据

                # 处理空数据
                if point_cloud.ndim == 1 or point_cloud.shape[1] != 5:
                    point_cloud = np.zeros((0, 5), dtype=np.float32)

                # 预处理（与训练时一致）
                if point_cloud.shape[0] > 0:
                    point_cloud = self.scaler.fit_transform(point_cloud)

                # 填充/截断
                if len(point_cloud) < self.max_points:
                    pad = np.zeros((self.max_points - len(point_cloud), 5))
                    point_cloud = np.concatenate([point_cloud, pad])
                else:
                    point_cloud = point_cloud[:self.max_points]

                point_clouds.append(point_cloud)

            # 处理帧数
            if len(point_clouds) == 0:
                return np.zeros((self.max_frames, self.max_points, 5)), np.zeros(self.max_frames)

            point_clouds = np.array(point_clouds)

            # 处理帧数（与你的handle_variable_frames逻辑一致）
            frame_num = len(point_clouds)
            if frame_num < self.max_frames:
                pad_frames = np.zeros((self.max_frames - frame_num, self.max_points, 5))
                point_clouds = np.concatenate([point_clouds, pad_frames])
            else:
                point_clouds = point_clouds[:self.max_frames]

            mask = np.ones(self.max_frames)
            mask[frame_num:] = 0


            return point_clouds, mask

        except Exception as e:
            raise RuntimeError(f"Error processing {filename}: {str(e)}")

    def predict(self, filename):
        """预测入口函数"""
        try:
            # 1. 数据预处理
            point_clouds, mask = self._process_single_csv(filename)

            # 2. 转换为Tensor
            pc_tensor = torch.tensor(point_clouds, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 3. 模型预测
            with torch.no_grad():
                if isinstance(self.model, HybridModel):
                    output = self.model(pc_tensor, mask_tensor)
                else:
                    output = self.model(pc_tensor)

            # 4. 计算概率
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = probabilities[0, 1].item()  # 假设类别1是跌倒

            # 5. 返回结果
            return {
                "prediction": 1 if confidence > 0.5 else 0,
                "confidence": round(confidence, 4),
                "class_description": "Fall detected" if confidence > 0.5 else "Normal activity"
            }

        except Exception as e:
            return {
                "error": str(e),
                "prediction": -1,
                "confidence": 0.0
            }


# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = Predictor(
        model_path="./best_model/111/best_model_epoch7_20250413_235216.pth",
        model_type="HybridModel",
        max_frames=40,
        max_points=100,
        method='mask'
    )



    # 设置要扫描的目录路径（请修改为您的目标目录）
    target_dir = "F:\\code\\rada\\script\\origin_data_to_csv\\fall_bed_data"

    # 获取目录下所有CSV文件
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))

    # 遍历预测每个文件
    for csv_path in csv_files:
        try:
            result = predictor.predict(csv_path)
            filename = os.path.basename(csv_path)

            # 格式化输出结果
            print(f"""
            {'=' * 40}
            文件名称: {filename}
            预测结果: {result['class_description']}
            置信度: {result['confidence'] * 100:.2f}%
            {'=' * 40}
            """)

        except Exception as e:
            print(f"处理文件 {csv_path} 时出错: {str(e)}")
