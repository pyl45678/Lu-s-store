import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict
import warnings
import pandas as pd
from tqdm import tqdm
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')


# ===================== 第一个模型：水动力系数预测 =====================

class ShipGeometryProcessor:
    """船舶几何特征处理器"""

    def __init__(self, grid_size: Tuple[int, int] = (32, 16)):
        self.grid_size = grid_size

    def hull_to_feature_map(self, hull_points: np.ndarray) -> np.ndarray:
        """将船体点云投影到2D特征图上"""
        if len(hull_points) == 0:
            return np.zeros(self.grid_size).flatten()

        x_min, x_max = hull_points[:, 0].min(), hull_points[:, 0].max()
        y_min, y_max = hull_points[:, 1].min(), hull_points[:, 1].max()

        ship_length = x_max - x_min
        ship_beam = y_max - y_min

        if ship_length == 0 or ship_beam == 0:
            return np.zeros(self.grid_size).flatten()

        x_normalized = (hull_points[:, 0] - x_min) / ship_length
        y_normalized = (hull_points[:, 1] - y_min) / ship_beam

        x_indices = np.clip((x_normalized * (self.grid_size[0] - 1)).astype(int),
                            0, self.grid_size[0] - 1)
        y_indices = np.clip((y_normalized * (self.grid_size[1] - 1)).astype(int),
                            0, self.grid_size[1] - 1)

        feature_map = np.zeros(self.grid_size)
        np.add.at(feature_map, (x_indices, y_indices), 1)

        if feature_map.max() > 0:
            feature_map /= feature_map.max()

        return feature_map.flatten()

    def extract_global_features(self, hull_points: np.ndarray) -> np.ndarray:
        """提取全局的水动力关键参数"""
        if len(hull_points) == 0:
            return np.zeros(12)

        x_min, x_max = hull_points[:, 0].min(), hull_points[:, 0].max()
        y_min, y_max = hull_points[:, 1].min(), hull_points[:, 1].max()
        z_min, z_max = hull_points[:, 2].min(), hull_points[:, 2].max()

        length = x_max - x_min
        beam = y_max - y_min
        draft = z_max - z_min

        # 基本几何参数
        volume_bbox = length * beam * draft
        displacement = volume_bbox * 0.6

        # 主要比率参数
        length_beam_ratio = length / beam if beam > 0 else 0
        beam_draft_ratio = beam / draft if draft > 0 else 0
        length_draft_ratio = length / draft if draft > 0 else 0
        block_coefficient = 0.6

        # 截面特性
        waterplane_area = length * beam * 0.8
        wetted_surface = 2 * (length * draft + beam * draft) + length * beam

        # 稳心和重心相关 
        gm_estimate = beam ** 2 / (12 * draft)
        kb_kg_ratio = 0.5

        global_features = np.array([
            length, beam, draft, displacement,
            length_beam_ratio, beam_draft_ratio, length_draft_ratio, block_coefficient,
            waterplane_area, wetted_surface, gm_estimate, kb_kg_ratio
        ])

        return global_features


class PhysicsConstrainedLayer(nn.Module):
    """物理约束层"""

    def __init__(self, matrix_dim: int = 6, matrix_type: str = 'spd', epsilon: float = 1e-4):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.matrix_type = matrix_type
        self.epsilon = epsilon
        self.n_tril_elements = matrix_dim * (matrix_dim + 1) // 2
        self.tril_indices = torch.tril_indices(row=matrix_dim, col=matrix_dim, offset=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if self.matrix_type == 'spd':
            L = torch.zeros(batch_size, self.matrix_dim, self.matrix_dim, device=x.device)
            L[:, self.tril_indices[0], self.tril_indices[1]] = x

            diag_indices = torch.arange(self.matrix_dim)
            L[:, diag_indices, diag_indices] = nn.functional.softplus(
                L[:, diag_indices, diag_indices]) + self.epsilon

            A = L @ L.transpose(-2, -1)
            return A
        else:
            matrix = torch.zeros(batch_size, self.matrix_dim, self.matrix_dim, device=x.device)
            matrix[:, self.tril_indices[0], self.tril_indices[1]] = x
            matrix = matrix + matrix.transpose(-2, -1)
            diag_mask = torch.eye(self.matrix_dim, device=x.device).bool()
            matrix[:, diag_mask] /= 2
            return matrix


class HydrodynamicPSM(nn.Module):
    """水动力系数预测模型（简化版，只预测阻尼矩阵用于横摇预测）"""

    def __init__(self, input_dim: int, matrix_dim: int = 6,
                 hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.n_output_elements = matrix_dim * (matrix_dim + 1) // 2

        # 构建网络
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.n_output_elements))
        self.network = nn.Sequential(*layers)

        # 物理约束层
        self.constraint = PhysicsConstrainedLayer(matrix_dim, 'spd')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        elements = self.network(x)
        damping_matrix = self.constraint(elements)
        return damping_matrix


# ===================== 第二个模型：横摇角预测 =====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.layer_norm(outputs)
        return outputs, hidden, cell


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        return output


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)


class IntegratedRollPredictionModel(nn.Module):
    """集成的横摇角预测模型"""

    def __init__(self, time_series_input_size, damping_matrix_size=36,
                 hidden_size=128, lstm_layers=2, transformer_layers=2,
                 nhead=8, dim_feedforward=256, forecast_steps=30, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.forecast_steps = forecast_steps
        self.damping_matrix_size = damping_matrix_size

        # 阻尼矩阵特征提取器
        self.damping_feature_extractor = nn.Sequential(
            nn.Linear(damping_matrix_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 融合层：将时间序列特征和阻尼矩阵特征结合
        combined_input_size = time_series_input_size + 32

        # LSTM编码器
        self.lstm_encoder = LSTMEncoder(
            input_size=combined_input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # Transformer解码器
        self.transformer_decoder = TransformerDecoderWrapper(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=transformer_layers,
            dropout=dropout
        )

        # 目标序列嵌入层
        self.target_embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU()
        )

        # 起始token
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # 输出层
        self.output_layer = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, time_series, damping_matrix, target=None, teacher_forcing_ratio=0.5):
        # time_series: [batch_size, seq_len, time_series_features]
        # damping_matrix: [batch_size, 36] (6x6 matrix flattened)
        # target: [batch_size, forecast_steps, 1]

        batch_size = time_series.size(0)
        seq_len = time_series.size(1)
        device = time_series.device

        # 提取阻尼矩阵特征
        damping_features = self.damping_feature_extractor(damping_matrix)
        # damping_features: [batch_size, 32]

        # 扩展阻尼特征到时间序列长度
        damping_features_expanded = damping_features.unsqueeze(1).repeat(1, seq_len, 1)
        # damping_features_expanded: [batch_size, seq_len, 32]

        # 融合时间序列特征和阻尼矩阵特征
        combined_input = torch.cat([time_series, damping_features_expanded], dim=2)
        # combined_input: [batch_size, seq_len, time_series_features + 32]

        # LSTM编码
        memory, _, _ = self.lstm_encoder(combined_input)

        # Transformer解码器掩码
        tgt_mask = generate_square_subsequent_mask(self.forecast_steps, device)

        # 初始化解码器输入
        start_token = self.start_token.repeat(batch_size, 1, 1)
        zeros = torch.zeros(batch_size, self.forecast_steps - 1, self.hidden_size, device=device)
        decoder_input = torch.cat([start_token, zeros], dim=1)

        if self.training and target is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # 教师强制
            embedded_target = self.target_embedding(target)
            decoder_input = torch.cat([start_token, embedded_target[:, :-1, :]], dim=1)

        # Transformer解码
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask
        )

        # 输出预测
        outputs = self.output_layer(decoder_output)

        return outputs


# ===================== 数据生成 =====================

def generate_ship_data(n_samples=2000):
    """生成船舶几何数据"""
    hull_data = []
    op_conditions = []

    for i in range(n_samples):
        # 生成船舶几何
        ship_types = ['container', 'tanker', 'bulk_carrier']
        ship_type = np.random.choice(ship_types)

        if ship_type == 'container':
            length = np.random.uniform(200, 400)
            beam = length / np.random.uniform(6, 8)
            draft = beam / np.random.uniform(2.2, 2.8)
        elif ship_type == 'tanker':
            length = np.random.uniform(250, 350)
            beam = length / np.random.uniform(5, 6.5)
            draft = beam / np.random.uniform(1.8, 2.3)
        else:
            length = np.random.uniform(100, 300)
            beam = length / np.random.uniform(5.5, 7.5)
            draft = beam / np.random.uniform(2.0, 3.0)

        # 生成船体点云
        n_points = np.random.randint(1000, 2000)
        x = np.random.uniform(-length / 2, length / 2, n_points)
        x_norm = x / (length / 2)
        beam_factor = (1 - x_norm ** 2) ** 0.5
        y_max = beam_factor * (beam / 2)
        y = np.random.uniform(-y_max, y_max)
        z = np.random.uniform(-draft, 0, n_points)
        hull_data.append(np.column_stack([x, y, z]))

        # 海况参数
        significant_wave_height = np.random.uniform(1.0, 8.0)
        peak_period = np.random.uniform(5.0, 15.0)
        wave_direction = np.random.uniform(0, 180)
        ship_speed = np.random.uniform(5, 20)
        encounter_frequency = 2 * np.pi / peak_period

        op_conditions.append([significant_wave_height, peak_period, wave_direction,
                              ship_speed, encounter_frequency])

    return hull_data, np.array(op_conditions)


def generate_damping_matrix(length, beam, draft, displacement, frequency, wave_height):
    """生成阻尼矩阵"""
    B = np.zeros((6, 6))
    omega = frequency

    # 对角元素
    B[0, 0] = 0.02 * displacement * omega
    B[1, 1] = 0.05 * displacement * omega
    B[2, 2] = 0.08 * displacement * omega
    B[3, 3] = 0.15 * displacement * beam ** 2 * omega  # 横摇阻尼
    B[4, 4] = 0.12 * displacement * length ** 2 * omega
    B[5, 5] = 0.08 * displacement * (length ** 2 + beam ** 2) * omega

    # 粘性阻尼影响
    viscous_factor = 1 + 0.5 * wave_height / beam
    B[1, 1] *= viscous_factor
    B[3, 3] *= viscous_factor * 2  # 横摇受粘性影响大

    # 耦合项
    B[1, 3] = B[3, 1] = 0.03 * displacement * beam * omega

    # 添加噪声
    B += np.random.normal(0, 0.1 * np.abs(B))

    # 确保正定
    B = 0.5 * (B + B.T)
    eigenvals = np.linalg.eigvals(B)
    if np.min(eigenvals) <= 0:
        B += (0.01 - np.min(eigenvals)) * np.eye(6)

    return B


def generate_time_series_data(damping_matrix, n_timesteps=100, forecast_steps=30):
    """基于阻尼矩阵生成横摇时间序列数据"""
    # 提取横摇相关的阻尼系数
    roll_damping = damping_matrix[3, 3]  # 横摇阻尼
    roll_sway_coupling = damping_matrix[1, 3]  # 横荡-横摇耦合

    # 模拟海浪激励
    t = np.linspace(0, 10, n_timesteps + forecast_steps)
    wave_freq = np.random.uniform(0.5, 1.5)
    wave_amplitude = np.random.uniform(0.5, 2.0)

    # 生成波浪激励
    wave_excitation = wave_amplitude * np.sin(wave_freq * t) + \
                      0.3 * wave_amplitude * np.sin(2 * wave_freq * t + np.pi / 4)

    # 生成船舶速度（相对稳定，有小幅变化）
    base_speed = np.random.uniform(10, 20)
    speed_variation = 0.5 * np.sin(0.1 * t) + 0.3 * np.random.randn(len(t)) * 0.1
    speed = base_speed + speed_variation
    speed = np.clip(speed, 5, 25)

    # 模拟横摇响应（基于阻尼系数）
    roll_natural_freq = np.random.uniform(0.8, 1.2)
    damping_ratio = min(roll_damping / (2 * 1000), 0.3)  # 归一化阻尼比

    # 二阶微分方程解（简化）
    roll_angle = np.zeros(len(t))
    roll_velocity = np.zeros(len(t))
    roll_acceleration = np.zeros(len(t))

    dt = t[1] - t[0]

    for i in range(1, len(t)):
        # 简化的横摇动力学方程
        external_moment = wave_excitation[i] * (1 + 0.1 * np.random.randn())
        damping_moment = -2 * damping_ratio * roll_natural_freq * roll_velocity[i - 1]
        restoring_moment = -(roll_natural_freq ** 2) * roll_angle[i - 1]

        roll_acceleration[i] = external_moment + damping_moment + restoring_moment
        roll_velocity[i] = roll_velocity[i - 1] + roll_acceleration[i] * dt
        roll_angle[i] = roll_angle[i - 1] + roll_velocity[i] * dt

        # 添加一些噪声
        roll_angle[i] += np.random.normal(0, 0.01)

    # 转换为度
    roll_angle_deg = np.rad2deg(roll_angle)

    roll_angle_deg = np.clip(roll_angle_deg, -10.0, 10.0)

    for i in range(1, len(roll_angle_deg)):
        if abs(roll_angle_deg[i] - roll_angle_deg[i - 1]) > 5.0:  # 如果变化过大
            roll_angle_deg[i] = roll_angle_deg[i - 1] + 0.3 * (roll_angle_deg[i] - roll_angle_deg[i - 1])


    # 构建特征矩阵 [speed, wave_excitation, roll_angle]
    features = np.column_stack([speed, wave_excitation, roll_angle_deg])

    return features


def create_integrated_dataset(n_samples=1500, time_step=30, forecast_step=30):
    """创建集成数据集"""
    print("生成集成数据集...")

    # 生成船舶数据
    hull_data, op_conditions = generate_ship_data(n_samples)

    # 第一步：准备水动力系数预测的数据
    geometry_processor = ShipGeometryProcessor()

    ship_features = []
    damping_matrices = []
    time_series_data = []
    roll_targets = []

    for i in tqdm(range(n_samples), desc="生成数据"):
        # 处理船舶几何特征
        feature_map = geometry_processor.hull_to_feature_map(hull_data[i])
        global_features = geometry_processor.extract_global_features(hull_data[i])
        combined_geometry = np.concatenate([feature_map, global_features])

        # 获取海况参数
        sea_conditions = op_conditions[i]
        ship_feature = np.concatenate([combined_geometry, sea_conditions])
        ship_features.append(ship_feature)

        # 生成阻尼矩阵
        hull_points = hull_data[i]
        x_min, x_max = hull_points[:, 0].min(), hull_points[:, 0].max()
        y_min, y_max = hull_points[:, 1].min(), hull_points[:, 1].max()
        z_min, z_max = hull_points[:, 2].min(), hull_points[:, 2].max()

        length = x_max - x_min
        beam = y_max - y_min
        draft = z_max - z_min
        displacement = length * beam * draft * 0.6 * 1.025

        damping_matrix = generate_damping_matrix(
            length, beam, draft, displacement,
            sea_conditions[4], sea_conditions[0]  # frequency, wave_height
        )
        damping_matrices.append(damping_matrix.flatten())

        # 生成时间序列数据
        total_steps = time_step + forecast_step
        time_series = generate_time_series_data(damping_matrix, total_steps)

        # 分离输入和目标
        input_series = time_series[:time_step]
        target_series = time_series[time_step:time_step + forecast_step, 2]  # 只取横摇角

        time_series_data.append(input_series)
        roll_targets.append(target_series)

    print(f"生成了 {n_samples} 个集成样本")

    return (np.array(ship_features), np.array(damping_matrices),
            np.array(time_series_data), np.array(roll_targets))


# ===================== 集成数据集类 =====================

class IntegratedDataset(Dataset):
    """集成数据集"""

    def __init__(self, ship_features, damping_matrices, time_series_data, roll_targets):
        self.ship_features = ship_features
        self.damping_matrices = damping_matrices
        self.time_series_data = time_series_data
        self.roll_targets = roll_targets

        # 标准化
        self.ship_scaler = StandardScaler()
        self.damping_scaler = StandardScaler()
        self.time_series_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self._preprocess()

    def _preprocess(self):
        # 标准化船舶特征
        self.scaled_ship_features = self.ship_scaler.fit_transform(self.ship_features)

        # 标准化阻尼矩阵
        self.scaled_damping = self.damping_scaler.fit_transform(self.damping_matrices)

        # 标准化时间序列数据
        n_samples, n_timesteps, n_features = self.time_series_data.shape
        reshaped_ts = self.time_series_data.reshape(-1, n_features)
        scaled_ts = self.time_series_scaler.fit_transform(reshaped_ts)
        self.scaled_time_series = scaled_ts.reshape(n_samples, n_timesteps, n_features)

        # 标准化目标数据
        self.scaled_targets = self.target_scaler.fit_transform(
            self.roll_targets.reshape(-1, 1)
        ).reshape(self.roll_targets.shape)

    def __len__(self):
        return len(self.ship_features)

    def __getitem__(self, idx):
        return {
            'ship_features': torch.FloatTensor(self.scaled_ship_features[idx]),
            'damping_matrix': torch.FloatTensor(self.scaled_damping[idx]),
            'time_series': torch.FloatTensor(self.scaled_time_series[idx]),
            'target': torch.FloatTensor(self.scaled_targets[idx])
        }


# ===================== 集成训练器 =====================

class IntegratedTrainer:
    """集成模型训练器"""

    def __init__(self, psm_model, roll_model, dataset, device='cuda'):
        self.psm_model = psm_model.to(device)
        self.roll_model = roll_model.to(device)
        self.dataset = dataset
        self.device = device

        # 分别为两个模型设置优化器
        self.psm_optimizer = optim.AdamW(psm_model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.roll_optimizer = optim.AdamW(roll_model.parameters(), lr=1e-4, weight_decay=1e-5)

        # 学习率调度器
        self.psm_scheduler = ReduceLROnPlateau(self.psm_optimizer, 'min', patience=10, factor=0.5)
        self.roll_scheduler = ReduceLROnPlateau(self.roll_optimizer, 'min', patience=10, factor=0.5)

        self.criterion = nn.MSELoss()
        self.history = {
            'psm_train': [], 'psm_val': [],
            'roll_train': [], 'roll_val': []
        }

    def train_step_1_psm(self, train_loader, val_loader, epochs=100):
        """第一阶段：训练PSM模型"""
        print("第一阶段：训练水动力系数预测模型...")

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.psm_model.train()
            train_loss = 0
            num_batches = 0

            for batch in train_loader:
                ship_features = batch['ship_features'].to(self.device)
                damping_true = batch['damping_matrix'].to(self.device)

                # 预测阻尼矩阵
                damping_pred = self.psm_model(ship_features)
                damping_pred_flat = damping_pred.view(damping_pred.size(0), -1)

                loss = self.criterion(damping_pred_flat, damping_true)

                self.psm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.psm_model.parameters(), 1.0)
                self.psm_optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches

            # 验证
            self.psm_model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    ship_features = batch['ship_features'].to(self.device)
                    damping_true = batch['damping_matrix'].to(self.device)

                    damping_pred = self.psm_model(ship_features)
                    damping_pred_flat = damping_pred.view(damping_pred.size(0), -1)

                    loss = self.criterion(damping_pred_flat, damping_true)
                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches

            self.history['psm_train'].append(avg_train_loss)
            self.history['psm_val'].append(avg_val_loss)

            self.psm_scheduler.step(avg_val_loss)

            print(f'PSM Epoch {epoch + 1:03d}/{epochs:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.psm_model.state_dict(), 'best_psm_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print("PSM模型早停")
                    break

        # 加载最佳PSM模型
        self.psm_model.load_state_dict(torch.load('best_psm_model.pth'))

    def train_step_2_roll(self, train_loader, val_loader, epochs=200):
        """第二阶段：训练横摇预测模型"""
        print("第二阶段：训练横摇角预测模型...")

        best_loss = float('inf')
        patience_counter = 0

        # 冻结PSM模型
        for param in self.psm_model.parameters():
            param.requires_grad = False

        for epoch in range(epochs):
            # 训练
            self.roll_model.train()
            self.psm_model.eval()
            train_loss = 0
            num_batches = 0

            for batch in train_loader:
                ship_features = batch['ship_features'].to(self.device)
                time_series = batch['time_series'].to(self.device)
                roll_target = batch['target'].to(self.device).unsqueeze(-1)

                # 使用PSM模型预测阻尼矩阵
                with torch.no_grad():
                    damping_pred = self.psm_model(ship_features)
                    damping_pred_flat = damping_pred.view(damping_pred.size(0), -1)

                # 预测横摇角
                roll_pred = self.roll_model(time_series, damping_pred_flat,
                                            roll_target, teacher_forcing_ratio=0.5)

                loss = self.criterion(roll_pred.squeeze(-1), roll_target.squeeze(-1))

                self.roll_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.roll_model.parameters(), 1.0)
                self.roll_optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches

            # 验证
            self.roll_model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    ship_features = batch['ship_features'].to(self.device)
                    time_series = batch['time_series'].to(self.device)
                    roll_target = batch['target'].to(self.device).unsqueeze(-1)

                    damping_pred = self.psm_model(ship_features)
                    damping_pred_flat = damping_pred.view(damping_pred.size(0), -1)

                    roll_pred = self.roll_model(time_series, damping_pred_flat)

                    loss = self.criterion(roll_pred.squeeze(-1), roll_target.squeeze(-1))
                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches

            self.history['roll_train'].append(avg_train_loss)
            self.history['roll_val'].append(avg_val_loss)

            self.roll_scheduler.step(avg_val_loss)

            print(f'Roll Epoch {epoch + 1:03d}/{epochs:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.roll_model.state_dict(), 'best_roll_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print("横摇模型早停")
                    break

        # 加载最佳横摇模型
        self.roll_model.load_state_dict(torch.load('best_roll_model.pth'))

    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # PSM训练曲线
        axes[0].plot(self.history['psm_train'], label='PSM Train', color='blue')
        axes[0].plot(self.history['psm_val'], label='PSM Val', color='red')
        axes[0].set_title('PSM Model Training')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Roll训练曲线
        axes[1].plot(self.history['roll_train'], label='Roll Train', color='green')
        axes[1].plot(self.history['roll_val'], label='Roll Val', color='orange')
        axes[1].set_title('Roll Prediction Model Training')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('integrated_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_integrated_model(psm_model, roll_model, test_loader, dataset, device):
    """评估集成模型"""
    psm_model.eval()
    roll_model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            ship_features = batch['ship_features'].to(device)
            time_series = batch['time_series'].to(device)
            roll_target = batch['target'].to(device)

            # PSM预测阻尼矩阵
            damping_pred = psm_model(ship_features)
            damping_pred_flat = damping_pred.view(damping_pred.size(0), -1)

            # 预测横摇角
            roll_pred = roll_model(time_series, damping_pred_flat)

            all_predictions.append(roll_pred.squeeze(-1).cpu().numpy())
            all_targets.append(roll_target.cpu().numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)



    print(f"预测形状: {predictions.shape}")
    print(f"目标形状: {targets.shape}")


    # 反标准化
    predictions_denorm = dataset.target_scaler.inverse_transform(
        predictions.reshape(-1, 1)).reshape(predictions.shape)
    targets_denorm = dataset.target_scaler.inverse_transform(
        targets.reshape(-1, 1)).reshape(targets.shape)

    # 计算指标
    mse = np.mean((predictions_denorm - targets_denorm) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_denorm.flatten(), predictions_denorm.flatten())
    r2 = r2_score(targets_denorm.flatten(), predictions_denorm.flatten())

    print(f"\n集成模型评估结果:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return predictions_denorm, targets_denorm


def plot_roll_predictions(predictions, targets, num_samples=4):
    """绘制横摇角预测结果"""
    sample_indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices[:4]):
        ax = axes[i]

        time_steps = np.arange(len(predictions[idx]))
        pred_seq = predictions[idx]
        true_seq = targets[idx]

        rmse = np.sqrt(np.mean((pred_seq - true_seq) ** 2))

        ax.plot(time_steps, true_seq, label='True', marker='o', color='blue', linewidth=2)
        ax.plot(time_steps, pred_seq, label='Predicted', marker='x',
                linestyle='--', color='red', linewidth=2)

        ax.set_title(f"Sample {idx} (RMSE: {rmse:.4f}°)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Roll Angle (degrees)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('integrated_roll_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


# ===================== 主程序 =====================

def main():
    """主执行函数"""
    print("=== 集成船舶横摇角预测系统 ===")

    # 设置参数
    time_step = 30
    forecast_step = 30
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 生成数据
    ship_features, damping_matrices, time_series_data, roll_targets = \
        create_integrated_dataset(n_samples=2000, time_step=time_step, forecast_step=forecast_step)

    # 创建数据集
    dataset = IntegratedDataset(ship_features, damping_matrices, time_series_data, roll_targets)

    # 数据划分
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")

    # 创建模型
    # PSM模型
    sample_ship_features = dataset.scaled_ship_features[0]
    psm_input_dim = len(sample_ship_features)
    psm_model = HydrodynamicPSM(input_dim=psm_input_dim, matrix_dim=6,
                                hidden_dims=[256, 128, 64])

    # 横摇预测模型
    time_series_input_size = 3  # [speed, wave_excitation, roll_angle]
    roll_model = IntegratedRollPredictionModel(
        time_series_input_size=time_series_input_size,
        damping_matrix_size=36,  # 6x6 matrix flattened
        hidden_size=128,
        lstm_layers=2,
        transformer_layers=2,
        nhead=8,
        dim_feedforward=256,
        forecast_steps=forecast_step,
        dropout=0.1
    )

    print(f"PSM模型参数: {sum(p.numel() for p in psm_model.parameters()):,}")
    print(f"横摇模型参数: {sum(p.numel() for p in roll_model.parameters()):,}")

    # 训练集成模型
    trainer = IntegratedTrainer(psm_model, roll_model, dataset, device)

    # 第一阶段：训练PSM模型
    trainer.train_step_1_psm(train_loader, val_loader, epochs=100)

    # 第二阶段：训练横摇预测模型
    trainer.train_step_2_roll(train_loader, val_loader, epochs=200)

    # 绘制训练历史
    trainer.plot_training_history()

    # 评估模型
    predictions, targets = evaluate_integrated_model(
        trainer.psm_model, trainer.roll_model, test_loader, dataset, device)

    # 可视化预测结果
    plot_roll_predictions(predictions, targets, num_samples=4)

    # 保存模型
    torch.save({
        'psm_state_dict': trainer.psm_model.state_dict(),
        'roll_state_dict': trainer.roll_model.state_dict(),
        'dataset_scalers': {
            'ship_scaler': dataset.ship_scaler,
            'damping_scaler': dataset.damping_scaler,
            'time_series_scaler': dataset.time_series_scaler,
            'target_scaler': dataset.target_scaler
        }
    }, 'integrated_model_complete.pth')

    print("集成模型训练完成！")


if __name__ == "__main__":
    main()