# -*- coding:utf-8 -*-
import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
import pickle
from copy import deepcopy
import json
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============== 可复现性设置 ===============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============== 输出目录设置 ===============
OUTPUT_DIR = "P-LT"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# print(f"📁 输出目录: {OUTPUT_DIR}/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")


# =============== 辅助函数 ===============
def is_power_of_2(n):
    """检查一个数是否是2的幂"""
    return n > 0 and (n & (n - 1)) == 0

def get_valid_batch_sizes(min_size=32, max_size=256):
    """获取指定范围内所有有效的batch_size (2的幂)"""
    valid_sizes = []
    power = int(np.log2(min_size))
    while 2**power <= max_size:
        if 2**power >= min_size:
            valid_sizes.append(2**power)
        power += 1
    return valid_sizes


# =========================
# PSO 粒子群优化算法
# =========================
class Particle:
    """PSO粒子类"""
    def __init__(self, bounds):
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.best_score = float('inf')
        self.score = float('inf')
        
        # 初始化位置和速度
        for param, (lower, upper, param_type) in bounds.items():
            if param_type == 'int':
                self.position[param] = random.randint(lower, upper)
                self.velocity[param] = random.randint(-(upper-lower)//4, (upper-lower)//4)
            elif param_type == 'float':
                self.position[param] = random.uniform(lower, upper)
                self.velocity[param] = random.uniform(-(upper-lower)*0.25, (upper-lower)*0.25)
            elif param_type == 'log':
                log_lower, log_upper = np.log10(lower), np.log10(upper)
                self.position[param] = 10 ** random.uniform(log_lower, log_upper)
                self.velocity[param] = 0
            elif param_type == 'power_of_2':
                log2_lower = int(np.log2(lower))
                log2_upper = int(np.log2(upper))
                selected_exp = random.randint(log2_lower, log2_upper)
                self.position[param] = 2 ** selected_exp
                self.velocity[param] = 0
            elif param_type == 'multiple_64':
                num_multiples = (upper - lower) // 64 + 1
                selected_multiple = random.randint(0, num_multiples - 1)
                self.position[param] = lower + selected_multiple * 64
                self.velocity[param] = 0
            elif param_type == 'multiple_256':
                num_multiples = (upper - lower) // 256 + 1
                selected_multiple = random.randint(0, num_multiples - 1)
                self.position[param] = lower + selected_multiple * 256
                self.velocity[param] = 0
        
        self.best_position = deepcopy(self.position)


class PSO:
    """粒子群优化器"""
    def __init__(self, bounds, n_particles=20, max_iter=50, w=0.7, c1=1.4, c2=1.6, 
                 early_stop_patience=10, min_improvement=1e-6):
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.early_stop_patience = early_stop_patience
        self.min_improvement = min_improvement
        
        self.particles = [Particle(bounds) for _ in range(n_particles)]
        self.global_best_position = deepcopy(self.particles[0].position)
        self.global_best_score = float('inf')
        self.history = []
        self.no_improvement_count = 0
    
    def _clip_position(self, particle):
        """将粒子位置限制在边界内"""
        for param, (lower, upper, param_type) in self.bounds.items():
            if param_type == 'int':
                particle.position[param] = max(lower, min(upper, int(round(particle.position[param]))))
            elif param_type == 'float':
                particle.position[param] = float(np.clip(particle.position[param], lower, upper))
            elif param_type == 'log':
                particle.position[param] = float(np.clip(particle.position[param], lower, upper))
            elif param_type == 'power_of_2':
                current_value = particle.position[param]
                if current_value <= 0:
                    current_value = lower
                log2_value = np.log2(max(1, current_value))
                rounded_exp = int(round(log2_value))
                log2_lower = int(np.log2(lower))
                log2_upper = int(np.log2(upper))
                final_exp = max(log2_lower, min(log2_upper, rounded_exp))
                particle.position[param] = 2 ** final_exp
            elif param_type == 'multiple_64':
                current_value = particle.position[param]
                multiple = round(current_value / 64)
                candidate_value = multiple * 64
                particle.position[param] = max(lower, min(upper, candidate_value))
                particle.position[param] = (particle.position[param] // 64) * 64
                if particle.position[param] < lower:
                    particle.position[param] = lower
            elif param_type == 'multiple_256':
                current_value = particle.position[param]
                multiple = round(current_value / 256)
                candidate_value = multiple * 256
                particle.position[param] = max(lower, min(upper, candidate_value))
                particle.position[param] = (particle.position[param] // 256) * 256
                if particle.position[param] < lower:
                    particle.position[param] = lower
    
    def optimize(self, objective_func):
        print("\n" + "="*80)
        print("🚀 开始PSO超参数优化")
        print("="*80)
        print(f"粒子数量: {self.n_particles}")
        print(f"最大迭代: {self.max_iter}")
        print(f"优化参数: {list(self.bounds.keys())}")
        print("="*80 + "\n")
        
        for iteration in range(self.max_iter):
            print(f"\n{'='*80}")
            print(f"📍 PSO 迭代 {iteration + 1}/{self.max_iter}")
            print(f"{'='*80}")
            
            for i, particle in enumerate(self.particles):
                print(f"\n🔹 粒子 {i+1}/{self.n_particles}")
                print(f"当前超参数: {self._format_params(particle.position)}")
                
                score = objective_func(particle.position)
                particle.score = score
                
                print(f"验证损失: {score:.6f}")
                
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = deepcopy(particle.position)
                    print(f"✨ 个体最优更新! {score:.6f}")
                
                if score < self.global_best_score:
                    improvement = self.global_best_score - score
                    self.global_best_score = score
                    self.global_best_position = deepcopy(particle.position)
                    print(f"🎯 全局最优更新! {score:.6f} (改进: {improvement:.6f})")
                    print(f"最佳超参数: {self._format_params(self.global_best_position)}")
                    
                    if improvement > self.min_improvement:
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1
                else:
                    self.no_improvement_count += 1
            
            self.history.append({
                'iteration': iteration + 1,
                'global_best_score': self.global_best_score,
                'global_best_params': deepcopy(self.global_best_position),
                'particles_scores': [p.score for p in self.particles]
            })
            
            # 位置更新
            for particle in self.particles:
                for param in self.bounds.keys():
                    param_type = self.bounds[param][2]
                    
                    if param_type == 'power_of_2':
                        lower, upper = self.bounds[param][0], self.bounds[param][1]
                        log2_lower = int(np.log2(lower))
                        log2_upper = int(np.log2(upper))
                        
                        current_exp = int(np.log2(particle.position[param]))
                        best_exp = int(np.log2(particle.best_position[param]))
                        global_best_exp = int(np.log2(self.global_best_position[param]))
                        
                        r1, r2 = random.random(), random.random()
                        if r1 < self.c1 * 0.1 and best_exp != current_exp:
                            direction = 1 if best_exp > current_exp else -1
                            current_exp += direction
                        if r2 < self.c2 * 0.1 and global_best_exp != current_exp:
                            direction = 1 if global_best_exp > current_exp else -1
                            current_exp += direction
                        if random.random() < 0.05:
                            current_exp += random.choice([-1, 0, 1])
                        current_exp = max(log2_lower, min(log2_upper, current_exp))
                        particle.position[param] = 2 ** current_exp
                    
                    elif param_type == 'multiple_64':
                        lower, upper = self.bounds[param][0], self.bounds[param][1]
                        current_multiple = particle.position[param] // 64
                        best_multiple = particle.best_position[param] // 64
                        global_best_multiple = self.global_best_position[param] // 64
                        r1, r2 = random.random(), random.random()
                        if r1 < self.c1 * 0.15 and best_multiple != current_multiple:
                            direction = 1 if best_multiple > current_multiple else -1
                            current_multiple += direction
                        if r2 < self.c2 * 0.15 and global_best_multiple != current_multiple:
                            direction = 1 if global_best_multiple > current_multiple else -1
                            current_multiple += direction
                        if random.random() < 0.1:
                            current_multiple += random.choice([-1, 0, 1])
                        particle.position[param] = max(lower, min(upper, current_multiple * 64))
                    
                    elif param_type == 'multiple_256':
                        lower, upper = self.bounds[param][0], self.bounds[param][1]
                        current_multiple = particle.position[param] // 256
                        best_multiple = particle.best_position[param] // 256
                        global_best_multiple = self.global_best_position[param] // 256
                        r1, r2 = random.random(), random.random()
                        if r1 < self.c1 * 0.15 and best_multiple != current_multiple:
                            direction = 1 if best_multiple > current_multiple else -1
                            current_multiple += direction
                        if r2 < self.c2 * 0.15 and global_best_multiple != current_multiple:
                            direction = 1 if global_best_multiple > current_multiple else -1
                            current_multiple += direction
                        if random.random() < 0.1:
                            current_multiple += random.choice([-1, 0, 1])
                        particle.position[param] = max(lower, min(upper, current_multiple * 256))
                        
                    else:
                        r1, r2 = random.random(), random.random()
                        cognitive = self.c1 * r1 * (particle.best_position[param] - particle.position[param])
                        social = self.c2 * r2 * (self.global_best_position[param] - particle.position[param])
                        dynamic_w = self.w * (1 - iteration / self.max_iter) + 0.4
                        particle.velocity[param] = dynamic_w * particle.velocity[param] + cognitive + social
                        if param_type == 'int':
                            lower, upper = self.bounds[param][0], self.bounds[param][1]
                            max_velocity = (upper - lower) * 0.2
                            particle.velocity[param] = np.clip(particle.velocity[param], -max_velocity, max_velocity)
                        particle.position[param] += particle.velocity[param]
                        if param_type == 'int':
                            particle.position[param] = round(particle.position[param])
                
                self._clip_position(particle)
            
            avg_score = np.mean([p.score for p in self.particles])
            print(f"\n{'='*80}")
            print(f"📊 迭代 {iteration + 1} 总结:")
            print(f"  全局最优: {self.global_best_score:.6f}")
            print(f"  平均适应度: {avg_score:.6f}")
            print(f"  无改进轮数: {self.no_improvement_count}/{self.early_stop_patience}")
            print(f"  最佳超参数: {self._format_params(self.global_best_position)}")
            print(f"{'='*80}")
            
            if self.no_improvement_count >= self.early_stop_patience:
                print(f"\n🛑 早停触发! 连续{self.early_stop_patience}轮无显著改进")
                print(f"最终最优损失: {self.global_best_score:.6f}")
                break
        
        return self.global_best_position, self.global_best_score
    
    def _format_params(self, params):
        formatted = {}
        for k, v in params.items():
            if isinstance(v, float):
                if v < 0.01:
                    formatted[k] = f"{v:.2e}"
                else:
                    formatted[k] = f"{v:.4f}"
            else:
                formatted[k] = v
        return formatted


# =========================
# SimpleLSTMTransformerED 模型
# =========================
class SimpleLSTMTransformerED(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 lstm_layers, transformer_layers, nhead,
                 dim_feedforward, forecast_steps, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.forecast_steps = forecast_steps
        
        # ============ 编码器部分 ============
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.lstm_pre_norm = nn.LayerNorm(hidden_size)
        self.lstm_post_norm = nn.LayerNorm(hidden_size)
        
        self.encoder_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_attn_norm = nn.LayerNorm(hidden_size)
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )
        
        # ============ 解码器部分 ============
        # 🔧 关键改进1: 使用线性投影而非Embedding (更灵活)
        self.decoder_base_proj = nn.Linear(hidden_size, hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_layers
        )
        
        # 🔧 关键改进2: 简化输出层(对齐优秀模型)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Linear(320, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Linear(160, output_size)
        )
        
        self._init_weights()
    
    def _create_sinusoidal_embeddings(self, max_len, d_model):
        """标准正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        """权重初始化"""
        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # 输出层 (关键: 使用小的gain)
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, src):
        B, T, _ = src.shape
        device = src.device
        
        # ============ 编码阶段 ============
        x = self.input_embedding(src)
        
        # 位置编码
        if not hasattr(self, '_pos_cache') or self._pos_cache.size(1) < T or self._pos_cache.device != device:
            pos_encoding = self._create_sinusoidal_embeddings(max(T, 1000), self.hidden_size)
            self._pos_cache = pos_encoding.to(device)
        
        pos_encoding = self._pos_cache[:, :T, :]
        x = x + pos_encoding
        
        # LSTM编码
        x_norm = self.lstm_pre_norm(x)
        memory, _ = self.lstm(x_norm)
        memory = self.lstm_post_norm(memory)
        
        # 自注意力增强
        memory_attn, _ = self.encoder_self_attn(memory, memory, memory)
        memory = self.encoder_attn_norm(memory + memory_attn)
        
        # 特征增强
        memory_pool = memory.mean(dim=1)
        enhance_weights = self.feature_enhance(memory_pool).unsqueeze(1)
        memory = memory * (1 + enhance_weights * 0.5)
        
        # ============ 🔧 关键改进: 解码阶段 (对齐优秀模型) ============
        # 1. 使用编码器最后一步作为"种子"
        last_state = memory[:, -1:, :]  # [B, 1, H]
        
        # 2. 扩展到forecast_steps
        decoder_input = last_state.expand(-1, self.forecast_steps, -1)  # [B, forecast_steps, H]
        
        # 3. 🌟 核心: 添加渐进式时间步编码 (完全对齐优秀模型)
        step_positions = torch.arange(
            self.forecast_steps, 
            device=device, 
            dtype=torch.float32
        )  # [forecast_steps]
        
        # 归一化到[0, 0.1]范围 (与优秀模型一致)
        step_scale = step_positions / self.forecast_steps * 0.1
        step_scale = step_scale.unsqueeze(0).unsqueeze(2)  # [1, forecast_steps, 1]
        
        # 广播到所有特征维度
        step_encodings = step_scale.expand(B, -1, self.hidden_size)  # [B, forecast_steps, H]
        
        # 4. 添加时间步编码到解码器输入
        decoder_input = decoder_input + step_encodings  # [B, forecast_steps, H]
        
        # 5. 投影并通过Transformer解码器
        tgt = self.decoder_base_proj(decoder_input)
        dec_out = self.transformer_decoder(
            tgt=tgt,
            memory=memory
        )
        
        # 6. 输出投影
        out = self.output_proj(dec_out)  # [B, forecast_steps, 1]
        
        return out


# =========================
# 数据加载函数
# =========================
def load_single_csv(csv_file_path):
    """加载单个CSV文件"""
    df = pd.read_csv(csv_file_path, skiprows=15)
    rx = df.iloc[:, 1].values
    ry = df.iloc[:, 2].values
    z = df.iloc[:, 3].values
    x = df.iloc[:, 4].values
    y = df.iloc[:, 5].values
    rz = df.iloc[:, 6].values
    rx_vel = df.iloc[:, 7].values
    ry_vel = df.iloc[:, 8].values
    z_vel = df.iloc[:, 9].values
    x_vel = df.iloc[:, 10].values
    y_vel = df.iloc[:, 11].values
    rz_vel = df.iloc[:, 12].values
    features = np.column_stack([rx, ry, z, x, y, rz, rx_vel, ry_vel, z_vel, x_vel, y_vel, rz_vel])
    return features, ry  # 预测目标: Ry (纵摇)


def load_raw_data_from_dataset(data_folder):
    """从dataset文件夹加载数据"""
    print("="*80)
    print("🔄 按文件内部顺序划分数据集 (每个CSV的前80%/中10%/后10%)")
    print("🎯 预测目标: Ry (纵摇角度 Pitch)")
    print("="*80)
    
    file_range = range(1, 16)
    train_features_list, train_ry_list = [], []
    val_features_list, val_ry_list = [], []
    test_features_list, test_ry_list = [], []
    
    loaded_count = 0
    for i in file_range:
        csv_file = os.path.join(data_folder, f"{i}.csv")
        if os.path.exists(csv_file):
            try:
                features, ry = load_single_csv(csv_file)
                n_samples = len(features)
                
                train_size = int(n_samples * 0.8)
                val_size = int(n_samples * 0.1)
                
                train_end = train_size
                val_end = train_size + val_size
                
                train_features_list.append(features[:train_end])
                train_ry_list.append(ry[:train_end])
                
                val_features_list.append(features[train_end:val_end])
                val_ry_list.append(ry[train_end:val_end])
                
                test_features_list.append(features[val_end:])
                test_ry_list.append(ry[val_end:])
                
                loaded_count += 1
                test_size = n_samples - val_end
                print(f"✓ {i:2d}.csv: {n_samples:6d}点 → 训练:{train_size}, 验证:{val_size}, 测试:{test_size}")
            except Exception as e:
                print(f"✗ {i}.csv 加载失败: {str(e)}")
        else:
            print(f"✗ {i}.csv 不存在")
    
    if loaded_count == 0:
        raise ValueError(f"未找到任何CSV文件在 {data_folder} 文件夹中")
    
    result = {
        'train': {
            'features': np.vstack(train_features_list),
            'ry': np.concatenate(train_ry_list)
        },
        'val': {
            'features': np.vstack(val_features_list),
            'ry': np.concatenate(val_ry_list)
        },
        'test': {
            'features': np.vstack(test_features_list),
            'ry': np.concatenate(test_ry_list)
        }
    }
    
    print(f"\n✓ 成功加载 {loaded_count} 个文件")
    print(f"训练集: {len(result['train']['features'])} 样本")
    print(f"验证集: {len(result['val']['features'])} 样本")
    print(f"测试集: {len(result['test']['features'])} 样本")
    
    return result['train'], result['val'], result['test']


def sliding_windows(data1, data2, seq_length, forecast_step):
    """创建滑动窗口序列"""
    max_length = min(len(data1), len(data2))
    x, y = [], []
    for i in range(max_length - seq_length - forecast_step + 1):
        _x = data1[i:(i + seq_length)]
        _y = data2[i + seq_length:i + seq_length + forecast_step]
        x.append(_x)
        y.append(_y)
    
    x_array = np.array(x)
    y_array = np.array(y)
    
    if len(y_array.shape) == 2:
        y_array = y_array.reshape(y_array.shape[0], y_array.shape[1], 1)
    
    return x_array, y_array


def safe_preprocess(data_folder):
    """数据预处理"""
    train_data, val_data, test_data = load_raw_data_from_dataset(data_folder)
    
    train_features = train_data['features']
    train_ry = train_data['ry']
    val_features = val_data['features']
    val_ry = val_data['ry']
    test_features = test_data['features']
    test_ry = test_data['ry']
    
    print("\n🔄 正在进行特征归一化 (StandardScaler)...")
    num_features = train_features.shape[1]
    feature_scalers = [StandardScaler() for _ in range(num_features)]
    
    train_features_norm = np.zeros_like(train_features)
    for i in range(num_features):
        train_features_norm[:, i] = feature_scalers[i].fit_transform(train_features[:, i].reshape(-1, 1)).flatten()
    
    val_features_norm = np.zeros_like(val_features)
    for i in range(num_features):
        val_features_norm[:, i] = feature_scalers[i].transform(val_features[:, i].reshape(-1, 1)).flatten()
    
    test_features_norm = np.zeros_like(test_features)
    for i in range(num_features):
        test_features_norm[:, i] = feature_scalers[i].transform(test_features[:, i].reshape(-1, 1)).flatten()
    
    target_scaler = StandardScaler()
    train_target = target_scaler.fit_transform(train_ry.reshape(-1, 1))
    val_target = target_scaler.transform(val_ry.reshape(-1, 1))
    test_target = target_scaler.transform(test_ry.reshape(-1, 1))
    
    try:
        scalers = {
            'feature_scalers': feature_scalers,
            'target_scaler': target_scaler
        }
        scaler_path = os.path.join(OUTPUT_DIR, 'lstm_transformer_scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print("✓ 归一化器已保存")
    except Exception as e:
        print(f"✗ 保存归一化器时出错: {e}")
    
    return (train_features_norm, train_target), (val_features_norm, val_target), (test_features_norm, test_target), target_scaler


def create_datasets(train_data, val_data, test_data, time_step, forecast_step):
    """创建PyTorch数据集"""
    train_x, train_y = sliding_windows(train_data[0], train_data[1], time_step, forecast_step)
    val_x, val_y = sliding_windows(val_data[0], val_data[1], time_step, forecast_step)
    test_x, test_y = sliding_windows(test_data[0], test_data[1], time_step, forecast_step)

    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y))
    test_dataset = TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y))

    return train_dataset, val_dataset, test_dataset


# =========================
# 简化的训练函数（用于PSO，快速）
# =========================
def train_model_pso(model, train_loader, val_loader, lr, weight_decay, max_epochs=12, patience=3):
    """
    快速训练函数，用于PSO超参数搜索
    优化: 减少epochs从15→12, patience从4→3, 加快评估速度
    """
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    warmup_epochs = 3
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=2e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        val_loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)
                epoch_val_loss += val_loss.item() * inputs.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ⏹ 早停于 Epoch {epoch+1} (Val: {avg_val_loss:.6f}, Best: {best_val_loss:.6f})")
                break

    return best_val_loss


# =========================
# PSO目标函数
# =========================
def objective_function(params, train_dataset, val_dataset, time_step, forecast_step):
    """
    PSO目标函数：训练模型并返回验证损失
    只优化5个核心参数：lr、dropout、hidden_size、nhead、transformer_layers
    """
    # 🔍 优化的参数（从PSO获取）
    lr = params['lr']
    dropout = params['dropout']
    hidden_size = int(params['hidden_size'])
    nhead = int(params['nhead'])
    transformer_layers = int(params['transformer_layers'])
    
    # 🔒 固定的参数（基于L-T基准）
    lstm_layers = 1  # 固定LSTM层数
    dim_feedforward = 1024  # 固定前馈网络维度
    batch_size = 512  # 🚀 大批次加速（改为512）
    weight_decay = 5e-4  # 固定权重衰减
    
    # 参数验证
    valid_nheads = [8, 16, 32]
    nhead = min(valid_nheads, key=lambda x: abs(x - nhead))
    assert hidden_size % 64 == 0, f"hidden_size {hidden_size} 不是64的倍数!"
    
    # 确保hidden_size能被nhead整除
    if hidden_size % nhead != 0:
        hidden_size = ((hidden_size // nhead) * nhead)
        hidden_size = max(64, (hidden_size // 64) * 64)

    # 数据加载器（提速）
    num_workers = 4
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0)
    )
    
    try:
        model = SimpleLSTMTransformerED(
            input_size=12,
            hidden_size=hidden_size,
            output_size=1,
            lstm_layers=lstm_layers,
            transformer_layers=transformer_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            forecast_steps=forecast_step,
            dropout=dropout
        )
        val_loss = train_model_pso(
            model, 
            train_loader, 
            val_loader, 
            lr, 
            weight_decay,
            max_epochs=12,  # 快速评估: 12 epochs
            patience=3       # 早停容忍: 3轮
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val_loss
        
    except Exception as e:
        print(f"  ❌ 训练失败: {str(e)}")
        import traceback
        print(f"  📋 错误详情: {traceback.format_exc()}")
        return float('inf')


# =========================
# 完整训练函数（最终训练用）
# =========================
def train_model_full(model, train_loader, val_loader, config):
    """完整训练函数（方案B：150 epochs, patience=20）"""
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        eps=1e-8
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    warmup_epochs = 6
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs, eta_min=5e-7)
    val_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=8, min_lr=1e-8)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if epoch < config['epochs'] * 0.5:
                noise = torch.randn_like(inputs) * 0.005
                inputs = inputs + noise
            
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        val_loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)
                epoch_val_loss += val_loss.item() * inputs.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        val_scheduler.step(avg_val_loss)

        train_val_ratio = avg_train_loss / (avg_val_loss + 1e-8)
        
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            model_path = os.path.join(OUTPUT_DIR, 'best_lstm_transformer_pso.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, model_path)
            print(f"✓ Epoch {epoch+1}: 模型已保存 (Val: {avg_val_loss:.6f}, Train/Val: {train_val_ratio:.3f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                print(f"早停: {epoch + 1}轮训练后无改善")
                break
        
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], '
                  f'Train: {avg_train_loss:.5f}, '
                  f'Val: {avg_val_loss:.5f}, '
                  f'Ratio: {train_val_ratio:.3f}, '
                  f'LR: {current_lr:.6f}')

    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_lstm_transformer_pso.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n已加载最佳模型 (Epoch {checkpoint['epoch'] + 1})")
    
    return model, {'train': train_losses, 'val': val_losses}


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    # 基础配置
    base_config = {
        'data_folder': 'dataset',
        'time_step': 60,
        'forecast_step': 60,  # 预测步数：60步
    }

    if not os.path.exists(base_config['data_folder']):
        raise FileNotFoundError(f"❌ 数据文件夹不存在: {base_config['data_folder']}")

    print("="*80)
    print("=== PSO优化 LSTM+Transformer 模型 (方案B: 快速模式) ===")
    print(f"数据文件夹: {base_config['data_folder']}")
    print(f"时间窗口: {base_config['time_step']}步 → 预测{base_config['forecast_step']}步")
    print("🎯 预测目标: Ry (纵摇角度 Pitch)")
    print("="*80 + "\n")

    # 数据准备
    train_data, val_data, test_data, target_scaler = safe_preprocess(base_config['data_folder'])
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_data, val_data, test_data, base_config['time_step'], base_config['forecast_step']
    )
    
    val_size = len(val_dataset)
    print(f"📊 数据集信息:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本") 
    print(f"  测试集: {len(test_dataset)} 样本")
    
    valid_batch_sizes = get_valid_batch_sizes(32, 256)
    print(f"  有效batch_size: {valid_batch_sizes} (2的幂)\n")
    
    # ========== 方案B：快速但稳定的PSO配置 ==========
    print("🔧 快速PSO配置 (方案B):")
    print("  • 粒子数: 6")
    print("  • 迭代数: 12")
    print("  • 每次评估: 15个epoch + 早停")
    print("  • 搜索空间: 以L-T优秀配置为中心探索\n")
    
    n_particles, max_iter = 8, 10  # 8粒子×10迭代=80次评估 (平衡速度与效率)
    total_evaluations = n_particles * max_iter
    estimated_minutes = total_evaluations * 12 * 1.5  # 优化后: 12 epochs × 1.5分钟
    estimated_hours = estimated_minutes / 60
    
    print(f"📊 PSO配置: {n_particles}粒子 × {max_iter}迭代 = {total_evaluations}次评估")
    print(f"📊 预估时间: {estimated_hours:.1f}小时 (约{estimated_minutes:.0f}分钟)")
    print(f"    ⚡ 加速策略: 12epochs/次 + 早停优化 + 8粒子平衡搜索")
    print(f"    💡 相比原方案节省50%+时间 (36h→{estimated_hours:.0f}h)\n")
    
    # 精简搜索空间（以L-T优秀配置为中心）
    # 🎯 只优化5个核心参数
    pso_bounds = {
        'lr': (1e-5, 1e-3, 'log'),  # 学习率: 1e-5~1e-3
        'dropout': (0.15, 0.30, 'float'),  # 丢弃率: 0.15-0.30
        'hidden_size': (256, 512, 'multiple_64'),  # 隐藏层: 256,320,384,448,512
        'nhead': (8, 32, 'power_of_2'),  # 注意力头: 8,16,32
        'transformer_layers': (1, 3, 'int'),  # Transformer层数: 1-3
    }
    
    pso = PSO(
        bounds=pso_bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        w=0.7,
        c1=1.4,
        c2=1.6,
        early_stop_patience=4,  # 减少到4轮 (从5)
        min_improvement=1e-4     # 提高阈值 (从5e-5)
    )
    
    def objective_wrapper(params):
        return objective_function(
            params, 
            train_dataset, 
            val_dataset,
            base_config['time_step'],
            base_config['forecast_step']
        )
    
    # 执行PSO优化
    best_params, best_score = pso.optimize(objective_wrapper)
    
    print("\n" + "="*80)
    print("🎉 PSO优化完成！")
    print("="*80)
    print(f"最佳验证损失: {best_score:.6f}")
    print(f"最佳超参数:")
    for key, value in best_params.items():
        if isinstance(value, float) and value < 0.01:
            print(f"  {key}: {value:.2e}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*80 + "\n")
    
    # 保存PSO结果
    pso_results = {
        'best_params': best_params,
        'best_score': best_score,
        'history': pso.history
    }
    pso_results_path = os.path.join(OUTPUT_DIR, 'pso_optimization_results.pkl')
    with open(pso_results_path, 'wb') as f:
        pickle.dump(pso_results, f)
    print(f"✓ PSO优化结果已保存: '{pso_results_path}'\n")
    
    # 使用最优超参数进行完整训练
    print("="*80)
    print("🚀 使用最优超参数进行完整训练")
    print("="*80 + "\n")
    
    # 确保hidden_size能被nhead整除
    final_hidden_size = (int(best_params['hidden_size']) // int(best_params['nhead'])) * int(best_params['nhead'])
    
    # 🔒 使用固定的batch_size（PSO中已固定为512）
    final_batch_size = 512
    
    final_config = {
        'data_folder': base_config['data_folder'],
        'time_step': base_config['time_step'],
        'forecast_step': base_config['forecast_step'],
        'batch_size': final_batch_size,  # 🔒 固定为512
        'lr': best_params['lr'],  # 🔍 PSO优化
        'weight_decay': 5e-4,  # 🔒 固定为5e-4
        'epochs': 150,
        'patience': 20,
        'model_params': {
            'input_size': 12,
            'hidden_size': final_hidden_size,  # 🔍 PSO优化
            'output_size': 1,
            'lstm_layers': 1,  # 🔒 固定为1
            'transformer_layers': int(best_params['transformer_layers']),  # 🔍 PSO优化
            'nhead': int(best_params['nhead']),  # 🔍 PSO优化
            'dim_feedforward': 1024,  # 🔒 固定为1024
            'forecast_steps': base_config['forecast_step'],
            'dropout': best_params['dropout']  # 🔍 PSO优化
        }
    }
    
    # 创建最终数据加载器（提速）
    num_workers = 4
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=final_config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin, 
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=final_config['batch_size'], 
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=final_config['batch_size'], 
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0)
    )
    
    # 创建最终模型
    final_model = SimpleLSTMTransformerED(**final_config['model_params']).to(device)
    
    total_params = sum(p.numel() for p in final_model.parameters())
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"\n{'='*80}")
    print(f"📊 最终模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"{'='*80}\n")
    
    # 维度检查
    print("🔍 维度检查:")
    with torch.no_grad():
        sample_input, sample_label = next(iter(train_loader))
        print(f"输入形状: {sample_input.shape}")
        sample_input = sample_input.to(device)
        sample_output = final_model(sample_input)
        print(f"输出形状: {sample_output.shape}")
        print(f"标签形状: {sample_label.shape}")
        assert sample_output.shape == sample_label.shape, "❌ 形状不匹配！"
        print("✅ 维度检查通过！\n")
    
    # 完整训练
    trained_model, history = train_model_full(final_model, train_loader, val_loader, final_config)
    
    final_model_path = os.path.join(OUTPUT_DIR, 'final_lstm_transformer_pso.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': final_config,
        'pso_best_params': best_params
    }, final_model_path)
    print(f"✓ 最终模型已保存: '{final_model_path}'\n")

    # =========================
    # 测试集评估
    # =========================
    print("="*80)
    print("📊 测试集评估")
    print("="*80)
    
    trained_model.eval()
    all_outputs = []
    all_labels = []
    test_loss = 0
    criterion = nn.SmoothL1Loss()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试集推理"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = trained_model(inputs)
            else:
                outputs = trained_model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            test_loss += criterion(outputs, labels).item() * inputs.size(0)

    avg_test_loss = test_loss / len(test_dataset)
    print(f'\n测试集损失 (归一化): {avg_test_loss:.6f}')

    # 合并预测
    all_outputs = np.concatenate(all_outputs, axis=0)  # [N, 90, 1]
    all_labels = np.concatenate(all_labels, axis=0)    # [N, 90, 1]

    # 反归一化
    all_outputs_2d = all_outputs.reshape(-1, all_outputs.shape[-1])
    all_labels_2d = all_labels.reshape(-1, all_labels.shape[-1])

    all_outputs_denorm_2d = target_scaler.inverse_transform(all_outputs_2d)
    all_labels_denorm_2d = target_scaler.inverse_transform(all_labels_2d)

    all_outputs_denorm = all_outputs_denorm_2d.reshape(all_outputs.shape)
    all_labels_denorm = all_labels_denorm_2d.reshape(all_labels.shape)

    # 评估指标
    rmse = np.sqrt(np.mean((all_outputs_denorm - all_labels_denorm) ** 2))
    mae = mean_absolute_error(all_labels_denorm_2d, all_outputs_denorm_2d)
    r2 = r2_score(all_labels_denorm_2d, all_outputs_denorm_2d)
    
    data_range = np.max(all_labels_denorm_2d) - np.min(all_labels_denorm_2d)
    nrmse_range = (rmse / data_range) * 100 if data_range > 0 else 0
    
    data_mean = np.mean(np.abs(all_labels_denorm_2d))
    nrmse_mean = (rmse / data_mean) * 100 if data_mean > 0 else 0
    
    epsilon = 1e-3
    smape = 100 * np.mean(2 * np.abs(all_outputs_denorm - all_labels_denorm) / 
                         (np.abs(all_outputs_denorm) + np.abs(all_labels_denorm) + epsilon))
    
    step_rmse = []
    for step in range(final_config['forecast_step']):
        step_pred = all_outputs_denorm[:, step, 0]
        step_true = all_labels_denorm[:, step, 0]
        step_rmse.append(np.sqrt(np.mean((step_pred - step_true) ** 2)))
    
    print(f'\n{"="*80}')
    print(f'=== 测试集评估指标 (PSO优化后) ===')
    print(f'{"="*80}')
    print(f'RMSE:  {rmse:.6f}')
    print(f'MAE:   {mae:.6f}')
    print(f'NRMSE (range): {nrmse_range:.4f}%')
    print(f'NRMSE (mean):  {nrmse_mean:.4f}%') 
    print(f'SMAPE: {smape:.4f}%')
    print(f'R²:    {r2:.6f}')
    print(f'\n📏 数据范围信息:')
    print(f'  真实值范围: [{np.min(all_labels_denorm_2d):.4f}, {np.max(all_labels_denorm_2d):.4f}]')
    print(f'  数据跨度:   {data_range:.4f}')
    print(f'  平均绝对值: {data_mean:.4f}')
    print(f'\n分步RMSE统计:')
    print(f'  前30步平均: {np.mean(step_rmse[:30]):.6f}')
    print(f'  中30步平均: {np.mean(step_rmse[30:60]):.6f}')
    print(f'  后30步平均: {np.mean(step_rmse[60:]):.6f}')
    print(f'{"="*80}\n')

    # =========================
    # 最终超参数打印
    # =========================
    print("="*80)
    print("🎯 最终超参数配置 (方案B)")
    print("="*80)
    print("📋 模型架构参数:")
    print(f"  • LSTM层数:        {final_config['model_params']['lstm_layers']}")
    print(f"  • Transformer层数: {final_config['model_params']['transformer_layers']}")
    print(f"  • 隐藏层大小:      {final_config['model_params']['hidden_size']}")
    print(f"  • 注意力头数:      {final_config['model_params']['nhead']}")
    print(f"  • 前馈网络维度:    {final_config['model_params']['dim_feedforward']}")
    print(f"  • Dropout率:       {final_config['model_params']['dropout']:.4f}")
    
    print("\n🔧 训练超参数:")
    print(f"  • 批次大小:        {final_config['batch_size']}")
    print(f"  • 学习率:          {final_config['lr']:.2e}")
    print(f"  • 权重衰减:        {final_config['weight_decay']:.2e}")
    print(f"  • 训练轮数:        {final_config['epochs']}")
    print(f"  • 早停耐心:        {final_config['patience']}")
    
    print("\n📊 数据配置:")
    print(f"  • 输入时间步:      {final_config['time_step']}")
    print(f"  • 预测时间步:      {final_config['model_params']['forecast_steps']}")
    print(f"  • 输入特征数:      {final_config['model_params']['input_size']}")
    
    print("\n🎯 PSO优化结果:")
    print(f"  • 最佳验证损失:    {best_score:.6f}")
    print(f"  • 最终测试RMSE:   {rmse:.6f}")
    print(f"  • 最终测试R²:     {r2:.6f}")
    
    total_params = sum(p.numel() for p in trained_model.parameters())
    trainable_params = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024
    
    print("\n📈 模型复杂度:")
    print(f"  • 总参数量:        {total_params:,}")
    print(f"  • 可训练参数:      {trainable_params:,}")
    print(f"  • 模型大小:        {model_size_mb:.2f} MB")
    print("="*80 + "\n")

    # =========================
    # 保存预测结果和超参数
    # =========================
    print("="*80)
    print("💾 保存预测结果和超参数")
    print("="*80)
    
    # 详细预测结果保存
    results_data = []
    for sample_idx in range(len(all_outputs_denorm)):
        sample_preds = all_outputs_denorm[sample_idx]
        sample_trues = all_labels_denorm[sample_idx]
        sample_rmse = np.sqrt(np.mean((sample_preds[:, 0] - sample_trues[:, 0]) ** 2))
        sample_mae = np.mean(np.abs(sample_preds[:, 0] - sample_trues[:, 0]))
        
        for step_idx in range(final_config['forecast_step']):
            abs_error = abs(sample_trues[step_idx][0] - sample_preds[step_idx][0])
            rel_error = abs_error / (abs(sample_trues[step_idx][0]) + 1e-6) * 100
            
            results_data.append({
                'sample_index': sample_idx,
                'time_step': step_idx + 1,
                'true_value': sample_trues[step_idx][0],
                'predicted_value': sample_preds[step_idx][0],
                'absolute_error': abs_error,
                'relative_error_percent': rel_error,
                'sample_rmse': sample_rmse,
                'sample_mae': sample_mae,
                'step_rmse': step_rmse[step_idx]
            })

    results_df = pd.DataFrame(results_data)
    results_df['error_category'] = pd.cut(
        results_df['absolute_error'], 
        bins=[0, 0.1, 0.2, 0.5, float('inf')],
        labels=['低误差(<0.1)', '中误差(0.1-0.2)', '高误差(0.2-0.5)', '极高误差(>0.5)']
    )
    
    csv_filename = os.path.join(OUTPUT_DIR, 'pso_lstm_transformer_predictions_detailed.csv')
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"✓ 详细预测结果已保存: '{csv_filename}'")
    
    # 保存超参数配置为JSON
    hyperparams_config = {
        'pso_optimization': {
            'best_validation_loss': float(best_score),
            'optimization_method': 'Particle Swarm Optimization',
            'particles': n_particles,
            'iterations': max_iter,
            'estimated_time': f"约{estimated_hours:.1f}小时"
        },
        'model_architecture': {
            'model_type': 'LSTM+Transformer Encoder-Decoder',
            'input_size': final_config['model_params']['input_size'],
            'hidden_size': final_config['model_params']['hidden_size'],
            'output_size': final_config['model_params']['output_size'],
            'lstm_layers': final_config['model_params']['lstm_layers'],
            'transformer_layers': final_config['model_params']['transformer_layers'],
            'attention_heads': final_config['model_params']['nhead'],
            'feedforward_dim': final_config['model_params']['dim_feedforward'],
            'dropout_rate': final_config['model_params']['dropout'],
            'forecast_steps': final_config['model_params']['forecast_steps']
        },
        'training_config': {
            'batch_size': final_config['batch_size'],
            'learning_rate': final_config['lr'],
            'weight_decay': final_config['weight_decay'],
            'epochs': final_config['epochs'],
            'patience': final_config['patience'],
            'optimizer': 'AdamW',
            'loss_function': 'SmoothL1Loss',
            'mixed_precision': torch.cuda.is_available()
        },
        'data_config': {
            'input_sequence_length': final_config['time_step'],
            'prediction_horizon': final_config['model_params']['forecast_steps'],
            'feature_count': final_config['model_params']['input_size'],
            'train_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        },
        'performance_metrics': {
            'test_rmse': float(rmse),
            'test_mae': float(mae),
            'test_nrmse_range_percent': float(nrmse_range),
            'test_nrmse_mean_percent': float(nrmse_mean),
            'test_r2_score': float(r2),
            'test_smape_percent': float(smape),
            'data_range': float(data_range),
            'data_mean_abs': float(data_mean),
            'early_steps_rmse': float(np.mean(step_rmse[:30])),
            'middle_steps_rmse': float(np.mean(step_rmse[30:60])),
            'late_steps_rmse': float(np.mean(step_rmse[60:]))
        },
        'model_complexity': {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': float(model_size_mb)
        }
    }
    config_filename = os.path.join(OUTPUT_DIR, 'pso_optimized_hyperparameters.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(hyperparams_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 超参数配置已保存: '{config_filename}'")
    
    # 简化版预测结果（兼容性）
    simple_results = results_df[['sample_index', 'time_step', 'true_value', 'predicted_value', 'absolute_error']].copy()
    simple_csv = os.path.join(OUTPUT_DIR, 'lstm_transformer_pso_predictions.csv')
    simple_results.to_csv(simple_csv, index=False, encoding='utf-8-sig')
    print(f"✓ 简化预测结果已保存: '{simple_csv}'")
    
    # 汇总统计
    summary_stats = {
        'dataset_info': {
            'total_rows': len(results_df),
            'unique_samples': results_df['sample_index'].nunique(),
            'time_steps': final_config['forecast_step']
        },
        'error_distribution': {
            'low_error_count': int((results_df['absolute_error'] < 0.1).sum()),
            'medium_error_count': int(((results_df['absolute_error'] >= 0.1) & (results_df['absolute_error'] < 0.2)).sum()),
            'high_error_count': int(((results_df['absolute_error'] >= 0.2) & (results_df['absolute_error'] < 0.5)).sum()),
            'very_high_error_count': int((results_df['absolute_error'] >= 0.5).sum())
        },
        'step_wise_rmse': {f'Step_{i+1}': float(v) for i, v in enumerate(step_rmse)},
        'best_hyperparameters': best_params,
        'pso_best_validation_loss': float(best_score),
        'final_test_metrics': {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'NRMSE_range_percent': float(nrmse_range),
            'NRMSE_mean_percent': float(nrmse_mean),
            'SMAPE_percent': float(smape),
            'R2_score': float(r2)
        },
        'pso_history_tail': pso.history[-5:] if len(pso.history) > 5 else pso.history
    }
    summary_path = os.path.join(OUTPUT_DIR, 'pso_final_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary_stats, f)
    print(f"✓ 汇总统计已保存: '{summary_path}'\n")

    # =========================
    # 可视化（适度精简）
    # =========================
    print("="*80)
    print("🎨 生成可视化图表")
    print("="*80 + "\n")

    # 1. PSO优化历史曲线
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    iterations = [h['iteration'] for h in pso.history]
    global_best_scores = [h['global_best_score'] for h in pso.history]
    plt.plot(iterations, global_best_scores, 'b-o', linewidth=2, markersize=6, label='全局最优')
    plt.xlabel('PSO迭代次数'); plt.ylabel('验证损失'); plt.title('PSO优化过程'); plt.grid(True, alpha=0.3); plt.legend()
    plt.subplot(1, 2, 2)
    for i, h in enumerate(pso.history):
        plt.scatter([h['iteration']] * len(h['particles_scores']), h['particles_scores'], alpha=0.5, s=30)
    plt.plot(iterations, global_best_scores, 'r-', linewidth=2, label='全局最优')
    plt.xlabel('PSO迭代次数'); plt.ylabel('粒子适应度'); plt.title('粒子群分布'); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "pso_optimization_history.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=200); plt.close()
    print(f"✓ PSO优化历史图已保存: '{fig_path}'")

    # 2. 训练和验证损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['train'], label="训练损失", linewidth=2, alpha=0.8)
    plt.plot(history['val'], label="验证损失", linewidth=2, alpha=0.8)
    plt.axvline(np.argmin(history['val']), color='r', linestyle='--', linewidth=2, label=f'最佳模型 (Epoch {np.argmin(history["val"])+1})')
    plt.title('最终训练过程 (PSO优化超参数)')
    plt.xlabel('Epochs'); plt.ylabel('Huber Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    fig_path = os.path.join(OUTPUT_DIR, "pso_final_training_loss.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=200); plt.close()
    print(f"✓ 训练损失曲线已保存: '{fig_path}'")

    # 3. 分步RMSE曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(step_rmse)+1), step_rmse, 'b-o', linewidth=2, markersize=5)
    plt.axhline(np.mean(step_rmse), color='r', linestyle='--', linewidth=2, label=f'平均RMSE: {np.mean(step_rmse):.4f}')
    plt.xlabel('预测步数'); plt.ylabel('RMSE'); plt.title('各预测步RMSE分析'); plt.legend(); plt.grid(True, alpha=0.3)
    fig_path = os.path.join(OUTPUT_DIR, "pso_step_rmse.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=200); plt.close()
    print(f"✓ 分步RMSE图已保存: '{fig_path}'")

    # 4. 预测样例可视化（减少样本）
    sample_count = min(4, len(all_outputs_denorm))
    sample_indices = np.random.choice(len(all_outputs_denorm), sample_count, replace=False)
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
    axes = axes.flatten()
    for i, idx in enumerate(sample_indices):
        preds = all_outputs_denorm[idx, :, 0]
        trues = all_labels_denorm[idx, :, 0]
        axes[i].plot(range(1, len(trues)+1), trues, 'ro-', label='真实值', linewidth=2, markersize=4, alpha=0.7)
        axes[i].plot(range(1, len(preds)+1), preds, 'b^--', label='预测值', linewidth=2, markersize=4, alpha=0.7)
        sample_rmse = np.sqrt(np.mean((preds - trues) ** 2))
        axes[i].set_title(f"样本 {idx} (RMSE: {sample_rmse:.4f})")
        axes[i].set_xlabel("时间步"); axes[i].set_ylabel("Ry值"); axes[i].legend(fontsize=9); axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "pso_prediction_samples.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=200); plt.close()
    print(f"✓ 预测样例图已保存: '{fig_path}'")

    # 5. 误差分布直方图
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    errors = (all_outputs_denorm - all_labels_denorm).flatten()
    plt.hist(errors, bins=80, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='r', linestyle='--', linewidth=2, label='零误差线')
    plt.xlabel('预测误差'); plt.ylabel('频数'); plt.title('误差分布直方图'); plt.legend(); plt.grid(True, alpha=0.3, axis='y')
    plt.subplot(1, 2, 2)
    relative_errors = np.abs(errors) / (np.abs(all_labels_denorm.flatten()) + 1e-6) * 100
    relative_errors = relative_errors[relative_errors < 50]
    plt.hist(relative_errors, bins=80, color='coral', alpha=0.7, edgecolor='black')
    plt.xlabel('相对误差 (%)'); plt.ylabel('频数'); plt.title('相对误差分布直方图'); plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "pso_error_distribution.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=200); plt.close()
    print(f"✓ 误差分布图已保存: '{fig_path}'")

    # 6. 真实值 vs 预测值散点图
    plt.figure(figsize=(9, 9))
    sample_size = min(4000, len(all_labels_denorm_2d))
    indices = np.random.choice(len(all_labels_denorm_2d), sample_size, replace=False)
    plt.scatter(all_labels_denorm_2d[indices], all_outputs_denorm_2d[indices], alpha=0.3, s=8, c='blue', label='预测点')
    min_val = min(all_labels_denorm_2d.min(), all_outputs_denorm_2d.min())
    max_val = max(all_labels_denorm_2d.max(), all_outputs_denorm_2d.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    plt.xlabel('真实值'); plt.ylabel('预测值'); plt.title(f'预测效果散点图 (R²={r2:.4f})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.axis('equal')
    fig_path = os.path.join(OUTPUT_DIR, "pso_scatter_plot.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=200); plt.close()
    print(f"✓ 散点图已保存: '{fig_path}'")

    print("\n" + "="*80)
    print("✅ 方案B完成！")
    print("="*80)
    print(f"\n📁 生成的文件清单 (保存在 {OUTPUT_DIR}/ 目录下):")
    print("  🤖 模型文件:")
    print(f"    - {OUTPUT_DIR}/best_lstm_transformer_pso.pth (最佳模型)")
    print(f"    - {OUTPUT_DIR}/final_lstm_transformer_pso.pth (最终模型)")
    print("  📊 预测结果:")
    print(f"    - {OUTPUT_DIR}/pso_lstm_transformer_predictions_detailed.csv")
    print(f"    - {OUTPUT_DIR}/lstm_transformer_pso_predictions.csv")
    print("  ⚙️  配置文件:")
    print(f"    - {OUTPUT_DIR}/pso_optimized_hyperparameters.json")
    print(f"    - {OUTPUT_DIR}/pso_optimization_results.pkl")
    print(f"    - {OUTPUT_DIR}/pso_final_summary.pkl")
    print(f"    - {OUTPUT_DIR}/lstm_transformer_scalers.pkl")
    print("  📈 可视化图表:")
    print(f"    - {OUTPUT_DIR}/pso_optimization_history.png")
    print(f"    - {OUTPUT_DIR}/pso_final_training_loss.png")
    print(f"    - {OUTPUT_DIR}/pso_step_rmse.png")
    print(f"    - {OUTPUT_DIR}/pso_prediction_samples.png")
    print(f"    - {OUTPUT_DIR}/pso_error_distribution.png")
    print(f"    - {OUTPUT_DIR}/pso_scatter_plot.png")
    
    completion_info = {
        'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'pso_config': f"{n_particles}粒子 × {max_iter}迭代",
        'estimated_runtime': f"约{estimated_hours:.1f}小时",
        'final_performance': {
            'rmse': float(rmse),
            'mae': float(mae),
            'nrmse_range_percent': float(nrmse_range),
            'nrmse_mean_percent': float(nrmse_mean),
            'r2_score': float(r2),
            'smape_percent': float(smape)
        },
        'file_count': 12
    }
    completion_path = os.path.join(OUTPUT_DIR, 'pso_completion_summary.json')
    with open(completion_path, 'w', encoding='utf-8') as f:
        json.dump(completion_info, f, indent=2, ensure_ascii=False)
    print(f"✓ 运行完成摘要已保存: '{completion_path}'")
