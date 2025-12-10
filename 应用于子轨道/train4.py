import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import random

# 添加当前目录到 sys.path，确保能导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===================== 模型导入（适配Model4） =====================
from model4 import KoopmanCVAE, loss_cvae_koopman
from dataset4 import load_orbital_file  

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ===================== Collate函数（修复核心问题） =====================
def collate_fn_unified(batch):
    """统一collate函数 - 修复维度处理和拼接问题"""
    Xi_list = []
    Xj_list = []
    delta_t_list = []
    file_id_list = []
    
    # 过滤无效数据
    valid_batch = [item for item in batch if item is not None and len(item) >= 3]
    if not valid_batch:
        # 返回空批次的默认值
        return (torch.zeros(0, 6), torch.zeros(0, 6), 
                torch.zeros(0, 1), torch.zeros(0, dtype=torch.long))
    
    for item in valid_batch:
        try:
            # 解析数据
            if len(item) == 4:
                Xi, Xj, delta_t, file_id = item
            elif len(item) == 3:
                Xi, Xj, delta_t = item
                file_id = 0
            else:
                continue
            
            # 强制转换为tensor并统一维度
            # Xi/Xj: 确保2D (1, 6)
            if isinstance(Xi, np.ndarray):
                Xi = torch.FloatTensor(Xi)
            if Xi.dim() == 1:
                Xi = Xi.unsqueeze(0)  # (6,) -> (1, 6)
            elif Xi.dim() > 2:
                Xi = Xi.view(-1, 6)[:1]  # 降维到(1,6)
            
            if isinstance(Xj, np.ndarray):
                Xj = torch.FloatTensor(Xj)
            if Xj.dim() == 1:
                Xj = Xj.unsqueeze(0)  # (6,) -> (1, 6)
            elif Xj.dim() > 2:
                Xj = Xj.view(-1, 6)[:1]
            
            # delta_t: 确保2D (1, 1)
            if isinstance(delta_t, (int, float, np.ndarray)):
                delta_t = torch.FloatTensor([delta_t]) if isinstance(delta_t, (int, float)) else torch.FloatTensor(delta_t)
            if delta_t.dim() == 0:
                delta_t = delta_t.unsqueeze(0).unsqueeze(0)  # () -> (1,1)
            elif delta_t.dim() == 1:
                delta_t = delta_t.unsqueeze(1)  # (1,) -> (1,1)
            
            # file_id: 确保1D (1,)
            file_id = torch.tensor(file_id, dtype=torch.long) if not isinstance(file_id, torch.Tensor) else file_id.long()
            if file_id.dim() == 0:
                file_id = file_id.unsqueeze(0)  # () -> (1,)
            
            # 添加到列表
            Xi_list.append(Xi)
            Xj_list.append(Xj)
            delta_t_list.append(delta_t)
            file_id_list.append(file_id)
            
        except Exception as e:
            print(f"处理批次项出错: {e}")
            continue
    
    # 处理空列表情况
    if not Xi_list:
        return (torch.zeros(0, 6), torch.zeros(0, 6), 
                torch.zeros(0, 1), torch.zeros(0, dtype=torch.long))
    
    # 拼接批次（确保最终维度：B×6, B×6, B×1, B）
    try:
        Xi_batch = torch.cat(Xi_list, dim=0).squeeze(1) if Xi_list[0].dim() == 2 else torch.cat(Xi_list, dim=0)
        Xj_batch = torch.cat(Xj_list, dim=0).squeeze(1) if Xj_list[0].dim() == 2 else torch.cat(Xj_list, dim=0)
        delta_t_batch = torch.cat(delta_t_list, dim=0)
        file_id_batch = torch.cat(file_id_list, dim=0).squeeze()  # 最终为1D (B,)
    except Exception as e:
        print(f"拼接批次出错: {e}")
        # 降级返回
        Xi_batch = torch.stack([x.squeeze(0) for x in Xi_list]) if Xi_list else torch.zeros(0,6)
        Xj_batch = torch.stack([x.squeeze(0) for x in Xj_list]) if Xj_list else torch.zeros(0,6)
        delta_t_batch = torch.stack([x.squeeze(0) for x in delta_t_list]) if delta_t_list else torch.zeros(0,1)
        file_id_batch = torch.stack([x.squeeze(0) for x in file_id_list]) if file_id_list else torch.zeros(0, dtype=torch.long)
    
    # 最终维度检查和修正
    Xi_batch = Xi_batch.view(-1, 6)  # 确保B×6
    Xj_batch = Xj_batch.view(-1, 6)  # 确保B×6
    delta_t_batch = delta_t_batch.view(-1, 1)  # 确保B×1
    file_id_batch = file_id_batch.view(-1)  # 确保B
    
    return Xi_batch, Xj_batch, delta_t_batch, file_id_batch

# ===================== Dataset类（增强鲁棒性） =====================
class StatePairDataset(Dataset):
    """统一的状态对Dataset类 - 增强鲁棒性"""
    def __init__(self, state_pairs):
        # 过滤无效数据
        self.state_pairs = []
        for pair in state_pairs:
            if pair is None or len(pair) < 3:
                continue
            # 确保状态数据是6维
            if len(pair[0]) != 6 or len(pair[1]) != 6:
                print(f"警告：跳过非6维状态对 {pair[:2]}")
                continue
            self.state_pairs.append(pair)
        
    def __len__(self):
        return len(self.state_pairs)
    
    def __getitem__(self, idx):
        try:
            pair = self.state_pairs[idx]
            
            # 解析数据
            Xi = pair[0]
            Xj = pair[1]
            delta_t = pair[2]
            file_id = pair[4] if len(pair) >=5 else (pair[3] if len(pair)>=4 else 0)
            
            # 强制转换为tensor并检查维度
            Xi = torch.FloatTensor(Xi) if isinstance(Xi, np.ndarray) else Xi
            Xj = torch.FloatTensor(Xj) if isinstance(Xj, np.ndarray) else Xj
            delta_t = torch.FloatTensor([delta_t])  # 1D (1,)
            file_id = torch.tensor(file_id, dtype=torch.long)  # 1D
            
            # 确保基础维度
            if Xi.dim() == 0:
                Xi = torch.zeros(6)
            elif Xi.dim() > 1:
                Xi = Xi.flatten()[:6]
            
            if Xj.dim() == 0:
                Xj = torch.zeros(6)
            elif Xj.dim() > 1:
                Xj = Xj.flatten()[:6]
            
            return Xi, Xj, delta_t, file_id
        
        except Exception as e:
            print(f"处理索引{idx}出错: {e}")
            # 返回安全默认值
            return (torch.zeros(6, dtype=torch.float32),
                    torch.zeros(6, dtype=torch.float32),
                    torch.zeros(1, dtype=torch.float32),
                    torch.tensor(0, dtype=torch.long))

# ===================== 核心功能：按文件ID预测 + 多文件融合 =====================
def predict_by_file_id(model, scaler, vis_train_data, vis_timestamps, file_id_mapping, pred_steps, device):
    """按文件ID单独预测，返回每个文件的预测结果"""
    file_pred_results = {}
    total_pred_traj = None
    total_true_traj = None
    
    model.eval()
    with torch.no_grad():
        for file_id, file_name in file_id_mapping.items():
            print(f"\n=== 预测文件ID {file_id}: {file_name} ===")
            # 1. 数据检查
            if file_id not in vis_train_data:
                print(f"⚠️ 文件ID {file_id} 无训练数据，跳过")
                continue
            
            file_data_dict = vis_train_data[file_id]
            # ===== 适配字典key: 'norm_states' =====
            if isinstance(file_data_dict, dict) and 'norm_states' in file_data_dict:
                file_data = file_data_dict['norm_states']  # 提取归一化状态数据
            else:
                print(f"⚠️ 文件ID {file_id} 无'norm_states'字段，跳过")
                continue
            
            # ===== 适配字典key: 'raw_timestamps' =====
            file_timestamps_dict = vis_timestamps.get(file_id, {})
            if isinstance(file_timestamps_dict, dict) and 'raw_timestamps' in file_timestamps_dict:
                file_timestamps = file_timestamps_dict['raw_timestamps']  # 提取原始时间戳
            else:
                print(f"⚠️ 文件ID {file_id} 无'raw_timestamps'字段，使用默认Δt")
                file_timestamps = []
            
            # 转换为tensor并确保设备
            if isinstance(file_data, np.ndarray):
                file_data = torch.FloatTensor(file_data)
            # 确保是tensor类型
            if not isinstance(file_data, torch.Tensor):
                print(f"⚠️ 文件ID {file_id} 的数据类型不支持: {type(file_data)}，跳过")
                continue
            file_data = file_data.to(device)
            
            # 初始状态 x0 (1, 6)
            if len(file_data) == 0:
                print(f"⚠️ 文件ID {file_id} 数据为空，跳过")
                continue
            x0_norm = file_data[0:1] if file_data.dim() >= 2 else file_data.unsqueeze(0)
            
            # 2. 构建Δt序列
            deltat_seq = []
            # 转换时间戳为numpy数组/列表（兼容tensor）
            if isinstance(file_timestamps, torch.Tensor):
                file_timestamps = file_timestamps.cpu().numpy()
            if isinstance(file_timestamps, (list, np.ndarray)) and len(file_timestamps) > 1:
                file_timestamps_np = np.array(file_timestamps)
                for i in range(min(pred_steps, len(file_timestamps_np)-1)):
                    dt = file_timestamps_np[i+1] - file_timestamps_np[i]
                    deltat_seq.append(dt)
                # 补齐到pred_steps
                if len(deltat_seq) < pred_steps:
                    dt_mean = np.mean(deltat_seq) if deltat_seq else 1.0
                    deltat_seq += [dt_mean] * (pred_steps - len(deltat_seq))
            else:
                deltat_seq = [1.0] * pred_steps
            
            deltat_seq = torch.FloatTensor(deltat_seq).unsqueeze(1).to(device)
            
            # 3. Model4预测（传入file_id）
            try:
                pred_traj_norm, _ = model.predict_long_term(
                    x0=x0_norm,
                    delta_t_seq=deltat_seq,
                    file_id=file_id,
                    steps=pred_steps,
                    use_prior=True
                )
            except Exception as e:
                print(f"预测文件ID {file_id} 出错: {e}")
                continue
            
            # 4. 反归一化
            if isinstance(pred_traj_norm, np.ndarray):
                pred_traj_norm_tensor = torch.tensor(pred_traj_norm, dtype=torch.float32).to(device)
            else:
                pred_traj_norm_tensor = pred_traj_norm.to(device)
            
            try:
                pred_traj_real = scaler.inverse_transform(pred_traj_norm_tensor).cpu().numpy()
            except:
                pred_traj_real = pred_traj_norm_tensor.cpu().numpy()
            
            # 5. 真实轨迹
            true_traj_norm = file_data[:pred_steps+1] if len(file_data)>=pred_steps+1 else file_data
            try:
                true_traj_real = scaler.inverse_transform(true_traj_norm).cpu().numpy()
            except:
                true_traj_real = true_traj_norm.cpu().numpy()
            
            # 6. 构建时间轴
            deltat_seq_real = deltat_seq.cpu().numpy().flatten()
            cumulative_time = np.cumsum(deltat_seq_real)
            time_axis = np.insert(cumulative_time, 0, 0.0)
            
            # 7. 存储结果
            file_pred_results[file_id] = (pred_traj_real, true_traj_real, time_axis, file_name)
            
            # 8. 累加预测结果
            if total_pred_traj is None:
                total_pred_traj = np.zeros_like(pred_traj_real)
                total_true_traj = np.zeros_like(true_traj_real)
            
            # 对齐长度后累加
            min_len = min(len(total_pred_traj), len(pred_traj_real))
            total_pred_traj[:min_len] += pred_traj_real[:min_len]
            
            true_min_len = min(len(total_true_traj), len(true_traj_real))
            total_true_traj[:true_min_len] += true_traj_real[:true_min_len]
    
    return file_pred_results, total_pred_traj, total_true_traj

def plot_file_predictions(file_pred_results, total_pred_traj, total_true_traj, vis_spec_data, save_dir):
    """绘制每个文件预测结果和总累加结果"""
    os.makedirs(os.path.join(save_dir, "file_predictions"), exist_ok=True)
    
    # 状态名称和颜色配置
    state_names = [
        'X Position (km)', 'Y Position (km)', 'Z Position (km)',
        'Vx Velocity (km/s)', 'Vy Velocity (km/s)', 'Vz Velocity (km/s)'
    ]
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    # 1. 绘制每个文件的预测结果
    for file_id, (pred_traj, true_traj, time_axis, file_name) in file_pred_results.items():
        try:
            fig = plt.figure(figsize=(22, 14))
            
            # 3D轨道图
            ax2 = fig.add_subplot(3, 3, 2, projection='3d')
            ax2.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], 
                    'k-', lw=1.5, alpha=0.6, label='Ground Truth')
            ax2.plot(pred_traj[:,0], pred_traj[:,1], pred_traj[:,2], 
                    'r--', lw=2.0, label='Koopman Prediction')
            ax2.scatter(true_traj[0,0], true_traj[0,1], true_traj[0,2], c='g', s=50, label='Start')
            ax2.set_xlabel('X (km)')
            ax2.set_ylabel('Y (km)')
            ax2.set_zlabel('Z (km)')
            ax2.set_title(f"File {file_id}: {file_name[:30]} (Prediction)")
            ax2.legend()
            
            # 6个状态量对比
            for i in range(6):
                ax = fig.add_subplot(3, 3, i+4)
                ax.plot(time_axis[:len(true_traj)], true_traj[:, i], color=colors[i], 
                        linestyle='-', lw=1.5, alpha=0.8, label='True Value')
                ax.plot(time_axis[:len(pred_traj)], pred_traj[:, i], color=colors[i], 
                        linestyle='--', lw=1.5, alpha=0.8, label='Pred Value')
                ax.set_title(f"{state_names[i]} - File {file_id}")
                ax.set_xlabel(f"Time (s) (Total: {time_axis[-1]:.1f}s)")
                ax.set_ylabel(state_names[i].split(' ')[0])
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            # 清理文件名中的非法字符
            safe_filename = "".join([c for c in file_name if c not in '<>:"/\\|?*'][:20])
            plot_path = os.path.join(save_dir, "file_predictions", f"file_{file_id}_{safe_filename}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ 文件{file_id}预测图已保存至: {plot_path}")
        except Exception as e:
            print(f"绘制文件{file_id}预测图出错: {e}")
            continue
    
    # 2. 绘制累加结果 vs vis_spec_data
    try:
        # 新增：检查总预测结果有效性
        if total_pred_traj is None or total_true_traj is None or len(file_pred_results) == 0:
            print("⚠️ 无有效累加预测结果，跳过总累加图绘制")
            return
        
        fig = plt.figure(figsize=(22, 14))
        
        # 3D总轨道图
        ax2 = fig.add_subplot(3, 3, 2, projection='3d')
        ax2.plot(total_true_traj[:,0], total_true_traj[:,1], total_true_traj[:,2], 
                'k-', lw=1.5, alpha=0.6, label='Total True (Sum)')
        ax2.plot(total_pred_traj[:,0], total_pred_traj[:,1], total_pred_traj[:,2], 
                'r--', lw=2.0, label='Total Pred (Sum)')
        
        # 添加vis_spec_data对比（修复：先转换为numpy数组，再切片）
        if vis_spec_data is not None:
            # 转换为numpy数组（兼容tensor）
            if isinstance(vis_spec_data, torch.Tensor):
                vis_spec_data_np = vis_spec_data.cpu().numpy()
            elif isinstance(vis_spec_data, list):
                vis_spec_data_np = np.array(vis_spec_data)
            else:
                vis_spec_data_np = vis_spec_data
            
            # 检查维度有效性
            if vis_spec_data_np.ndim >= 2 and len(vis_spec_data_np) >= len(total_pred_traj):
                ax2.plot(vis_spec_data_np[:len(total_pred_traj),0], 
                        vis_spec_data_np[:len(total_pred_traj),1], 
                        vis_spec_data_np[:len(total_pred_traj),2], 
                        'b:', lw=2.5, label='vis_spec_data (True)')
            else:
                print(f"⚠️ vis_spec_data维度不匹配，跳过绘制: {vis_spec_data_np.shape}")
        
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        ax2.set_zlabel('Z (km)')
        ax2.set_title(f"Total Prediction (All Files Sum)")
        ax2.legend()
        
        # 6个状态量累加对比
        for i in range(6):
            ax = fig.add_subplot(3, 3, i+4)
            ax.plot(total_true_traj[:, i], color=colors[i], 
                    linestyle='-', lw=1.5, alpha=0.8, label='Total True (Sum)')
            ax.plot(total_pred_traj[:, i], color=colors[i], 
                    linestyle='--', lw=1.5, alpha=0.8, label='Total Pred (Sum)')
            
            # 添加vis_spec_data（修复：同样先转换为numpy数组）
            if vis_spec_data is not None:
                if isinstance(vis_spec_data, torch.Tensor):
                    vis_spec_data_np = vis_spec_data.cpu().numpy()
                elif isinstance(vis_spec_data, list):
                    vis_spec_data_np = np.array(vis_spec_data)
                else:
                    vis_spec_data_np = vis_spec_data
                
                if vis_spec_data_np.ndim >= 2 and len(vis_spec_data_np) >= len(total_pred_traj):
                    ax.plot(vis_spec_data_np[:len(total_pred_traj), i], color='black', 
                            linestyle=':', lw=1.5, alpha=0.8, label='vis_spec_data')
            
            ax.set_title(f"{state_names[i]} (Total Sum)")
            ax.set_xlabel("Step")
            ax.set_ylabel(state_names[i].split(' ')[0])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        total_plot_path = os.path.join(save_dir, "total_prediction_sum.png")
        plt.savefig(total_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 总累加预测图已保存至: {total_plot_path}")
    except Exception as e:
        print(f"绘制总累加预测图出错: {e}")
        import traceback
        traceback.print_exc()

# ===================== 封装预测+绘图 =====================
def predict_and_plot(model, scaler, full_data, full_timestamps, file_id, pred_steps, device, save_dir, epoch_num):
    """适配Model4的阶段性预测绘图"""
    try:
        print(f"\n=== Epoch {epoch_num} 阶段性预测与绘图（File ID: {file_id}）===")
        # 1. 数据准备
        full_data = full_data.to(device) if isinstance(full_data, torch.Tensor) else torch.FloatTensor(full_data).to(device)
        x0_norm = full_data[0:1] if full_data.dim() >=2 else full_data.unsqueeze(0)
        
        # 2. 构建Δt序列
        deltat_seq = []
        if full_timestamps is not None and len(full_timestamps) > 1:
            for i in range(min(pred_steps, len(full_timestamps)-1)):
                dt = full_timestamps[i+1] - full_timestamps[i]
                deltat_seq.append(dt)
            if len(deltat_seq) < pred_steps:
                dt_mean = np.mean(deltat_seq) if deltat_seq else 1.0
                deltat_seq += [dt_mean] * (pred_steps - len(deltat_seq))
        else:
            deltat_seq = [1.0] * pred_steps
        
        deltat_seq = torch.FloatTensor(deltat_seq).unsqueeze(1).to(device)

        # 3. 预测
        model.eval()
        with torch.no_grad():
            pred_traj_norm, _ = model.predict_long_term(
                x0=x0_norm,
                delta_t_seq=deltat_seq,
                file_id=file_id,
                steps=pred_steps,
                use_prior=True
            )
        
        # 4. 反归一化
        pred_traj_norm_tensor = torch.tensor(pred_traj_norm, dtype=torch.float32).to(device) if isinstance(pred_traj_norm, np.ndarray) else pred_traj_norm
        try:
            pred_traj_real = scaler.inverse_transform(pred_traj_norm_tensor).cpu().numpy()
        except:
            pred_traj_real = pred_traj_norm_tensor.cpu().numpy()
        
        # 5. 真实轨迹
        true_traj_norm = full_data[:pred_steps+1] if len(full_data)>=pred_steps+1 else full_data
        try:
            true_traj_real = scaler.inverse_transform(true_traj_norm).cpu().numpy()
        except:
            true_traj_real = true_traj_norm.cpu().numpy()
        
        # 6. 时间轴
        deltat_seq_real = deltat_seq.cpu().numpy().flatten()
        cumulative_time = np.cumsum(deltat_seq_real)
        time_axis = np.insert(cumulative_time, 0, 0.0)
        total_time = cumulative_time[-1] if len(cumulative_time) > 0 else 0.0
        
        # 7. 绘图
        fig = plt.figure(figsize=(22, 14))

        # Loss曲线
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.plot(loss_history[:epoch_num], color='navy', label=f'Total Loss (Epoch {epoch_num})')
        ax1.set_title(f"Training Loss (Epoch {epoch_num})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 3D轨道图
        ax2 = fig.add_subplot(3, 3, 2, projection='3d')
        ax2.plot(true_traj_real[:,0], true_traj_real[:,1], true_traj_real[:,2], 
                'k-', lw=1.5, alpha=0.6, label='Ground Truth')
        ax2.plot(pred_traj_real[:,0], pred_traj_real[:,1], pred_traj_real[:,2], 
                'r--', lw=2.0, label='Koopman Prediction')
        ax2.scatter(true_traj_real[0,0], true_traj_real[0,1], true_traj_real[0,2], c='g', s=50, label='Start')
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        ax2.set_zlabel('Z (km)')
        ax2.set_title(f"Orbital Prediction (File ID {file_id}, Epoch {epoch_num})")
        ax2.legend()

        # 6个状态量
        state_names = [
            'X Position (km)', 'Y Position (km)', 'Z Position (km)',
            'Vx Velocity (km/s)', 'Vy Velocity (km/s)', 'Vz Velocity (km/s)'
        ]
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

        for i in range(6):
            ax = fig.add_subplot(3, 3, i+4)
            ax.plot(time_axis[:len(true_traj_real)], true_traj_real[:, i], color=colors[i], 
                    linestyle='-', lw=1.5, alpha=0.8, label='True Value')
            ax.plot(time_axis[:len(pred_traj_real)], pred_traj_real[:, i], color=colors[i], 
                    linestyle='--', lw=1.5, alpha=0.8, label='Pred Value')
            ax.set_title(f"{state_names[i]} (Epoch {epoch_num})")
            ax.set_xlabel(f"Time (s) (Total: {total_time:.1f}s)")
            ax.set_ylabel(state_names[i].split(' ')[0])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 8. 保存
        plot_filename = f"prediction_file_{file_id}_epoch_{epoch_num}.png"
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Epoch {epoch_num} 预测图已保存至: {plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"阶段性预测绘图出错: {e}")
    finally:
        model.train()

# ===================== 主训练函数 =====================
def run_training_and_viz():
    # 配置
    data_dir = "data" 
    if os.path.exists(data_dir):
        FILENAME = data_dir
    else:
        filename_base = "轨道1-周期90.0min-半长轴6652.6km-e0.000-倾角90.0°-升交点0.0°-近地点0.0°.txt"
        possible_paths = [os.path.join("data", filename_base), filename_base]
        FILENAME = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if FILENAME is None:
        print(f"错误: 找不到数据文件或目录")
        return

    # 训练参数
    STATE_DIM = 6
    OBS_DIM = 64
    K_LATENT_DIM = 32
    BATCH_SIZE = 64
    EPOCHS = 100
    LR_BASE = 1e-4
    LR_DECAY = 1e-5
    LAMBDA_K_PRODUCT = 10.0
    NON_ADJ_RATIO = 1
    
    PRED_STEPS = 2000
    PLOT_INTERVAL = 10
    DELTA = 1.0
    MAX_FILE_ID = 100

        # ===================== 主训练函数中1. 数据加载部分 修改 =====================
    # 1. 数据加载
    print("加载轨道数据并生成状态对...")
    try:
        # 修复：正确接收load_orbital_file的6个返回值
        adj_pairs, non_adj_pairs, scaler, vis_spec_data_dict, vis_train_data_list, file_mapping = load_orbital_file(
            FILENAME, non_adj_ratio=NON_ADJ_RATIO
        )
        # 备份full_data/full_timestamps（兼容原有阶段性绘图逻辑）
        full_data = vis_spec_data_dict['norm_states']
        full_timestamps = vis_spec_data_dict['raw_timestamps']
    except Exception as e:
        print(f"加载数据出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 合并状态对
    all_pairs = []
    if non_adj_pairs and len(non_adj_pairs) > 0:
        all_pairs.extend(non_adj_pairs)
    
    if len(all_pairs) == 0:
        print("无法继续：无有效状态对。")
        return

    # 构建DataLoader
    try:
        unified_dataset = StatePairDataset(all_pairs)
        unified_dataloader = DataLoader(
            unified_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=collate_fn_unified,
            drop_last=True,
            num_workers=0,  # 禁用多进程加载（Windows下避免错误）
            pin_memory=False
        )
    except Exception as e:
        print(f"构建DataLoader出错: {e}")
        return
    
    print(f"训练数据统计：")
    print(f"  总状态对数量: {len(unified_dataset)}")
    print(f"  文件编号映射: {file_mapping}")
    
    # 2. Model4初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = KoopmanCVAE(
            state_dim=STATE_DIM,
            obs_dim=OBS_DIM,
            k_latent_dim=K_LATENT_DIM,
            max_file_id=MAX_FILE_ID
        ).to(device)
    except Exception as e:
        print(f"初始化模型出错: {e}")
        return
    
    optimizer = optim.Adam(model.parameters(), lr=LR_BASE, weight_decay=1e-5)
    
    # 学习率调度
    def lr_lambda(epoch):
        return 1.0 if epoch < 50 else 0.1
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 3. 训练循环
    print("开始训练Model4（file_id+Δt标签）...")
    global loss_history
    loss_history = []
    save_dir = "results_train4"
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} | 当前学习率: {current_lr:.1e}")
        
        total_epoch_loss = 0
        total_epoch_psi_loss = 0
        total_epoch_x_loss = 0
        total_epoch_kl_loss = 0
        total_epoch_latent_loss = 0
        batch_num = 0
        
        progress = tqdm(unified_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in progress:
            try:
                # 批次数据处理
                Xi_batch, Xj_batch, delta_t_batch, file_id_batch = batch
                
                # 跳过空批次
                if Xi_batch.shape[0] == 0:
                    continue
                
                # 转移到设备
                Xi_batch = Xi_batch.to(device)
                Xj_batch = Xj_batch.to(device)
                delta_t_batch = delta_t_batch.to(device)
                file_id_batch = file_id_batch.to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                psi_i, psi_j, K_matrix, psi_recon, x_i_recon, x_j_recon, mu, log_var = model.forward_single_pair(
                    Xi_batch, Xj_batch, delta_t_batch, file_id_batch
                )
                
                # 损失计算
                loss, psi_loss, x_loss, kl_loss, k_product_loss, latent_loss = loss_cvae_koopman(
                    psi_recon, psi_j,
                    x_i_recon, Xi_batch,
                    x_j_recon, Xj_batch,
                    mu, log_var, K_matrix,
                    file_ids=file_id_batch,
                    lambda_k_product=LAMBDA_K_PRODUCT,
                    delta=DELTA
                )
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # 累计损失
                total_epoch_loss += loss.item()
                total_epoch_psi_loss += psi_loss.item()
                total_epoch_x_loss += x_loss.item()
                total_epoch_kl_loss += kl_loss.item()
                total_epoch_latent_loss += latent_loss.item()
                batch_num += 1
                
                progress.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'latent_loss': f"{latent_loss.item():.6f}"
                })
            except Exception as e:
                print(f"\n批次处理出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算平均损失
        if batch_num > 0:
            avg_loss = total_epoch_loss / batch_num
            avg_psi_loss = total_epoch_psi_loss / batch_num
            avg_x_loss = total_epoch_x_loss / batch_num
            avg_kl_loss = total_epoch_kl_loss / batch_num
            avg_latent_loss = total_epoch_latent_loss / batch_num
        else:
            avg_loss = 0
            avg_psi_loss = 0
            avg_x_loss = 0
            avg_kl_loss = 0
            avg_latent_loss = 0

        loss_history.append(avg_loss)
        print(f"\nEpoch {epoch + 1}: ")
        print(f"  Average Loss = {avg_loss:.6f}(psi_loss={avg_psi_loss:.6f}, x_loss={avg_x_loss:.6f}, kl_loss={avg_kl_loss:.6f}, latent_loss={avg_latent_loss:.6f})")

        # 更新学习率
        lr_scheduler.step()
        """""
        # 阶段性绘图
        current_epoch = epoch + 1
        if current_epoch % PLOT_INTERVAL == 0 and file_mapping and len(file_mapping) > 0:
            sample_file_id = next(iter(file_mapping.keys()))
            predict_and_plot(
                model=model,
                scaler=scaler,
                full_data=full_data,
                full_timestamps=full_timestamps,
                file_id=sample_file_id,
                pred_steps=min(PRED_STEPS, 2000),  # 减少绘图步数加快速度
                device=device,
                save_dir=save_dir,
                epoch_num=current_epoch
            )
            """

    # 4. 保存最终模型
    try:
        model_path = os.path.join(save_dir, "koopman_cvae_model4_final.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\n最终Model4模型已保存至: {model_path}")
    except Exception as e:
        print(f"保存模型出错: {e}")

    # ===================== 主训练函数中5. 按文件ID预测部分 修改 =====================
    # 5. 按文件ID预测 + 多文件融合
    print("\n=== 开始按文件ID预测并融合结果 ===")
    # 构建按文件ID划分的训练数据（修复核心：每个file_id对应自己的训练数据）
    vis_train_data = {}
    vis_timestamps = {}

    # 关键修复：从load_orbital_file返回的vis_train_data（第5个返回值）中提取每个file_id的真实数据
    # 原load_orbital_file返回值：adj_pairs, non_adj_pairs, scaler, vis_spec_data, vis_train_data_list, file_mapping
    # 因此需要先接收正确的返回值：
    adj_pairs, non_adj_pairs, scaler, vis_spec_data_dict, vis_train_data_list, file_mapping = load_orbital_file(
        FILENAME, non_adj_ratio=NON_ADJ_RATIO
    )

    # 遍历vis_train_data_list（每个元素是单个文件的可视化格式字典），按file_id映射
    for file_data_dict in vis_train_data_list:
        fid = file_data_dict['file_id']
        vis_train_data[fid] = file_data_dict  # 每个file_id对应自己的真实数据字典
        vis_timestamps[fid] = file_data_dict  # timestamps也从该字典提取

    # 修复vis_spec_data：提取字典中的norm_states（而非整个字典）
    #vis_spec_data = vis_spec_data_dict['norm_states']  # 可视化数据的状态量tensor
    # 修复：vis_spec_data 反归一化（转为真实值）
    vis_spec_data = vis_spec_data_dict['norm_states']
    # 反归一化（关键！）
    vis_spec_data_real = scaler.inverse_transform(vis_spec_data).cpu().numpy()
    # 替换原 vis_spec_data 为反归一化后的真实值
    vis_spec_data = vis_spec_data_real

    # 预测
    try:
        file_pred_results, total_pred_traj, total_true_traj = predict_by_file_id(
            model=model,
            scaler=scaler,
            vis_train_data=vis_train_data,
            vis_timestamps=vis_timestamps,
            file_id_mapping=file_mapping,
            pred_steps=min(PRED_STEPS, 2000),  # 减少预测步数加快速度
            device=device
        )
        
        # 可视化
        plot_file_predictions(file_pred_results, total_pred_traj, total_true_traj, vis_spec_data, save_dir)
        
        # 保存预测数据
        np.savez(os.path.join(save_dir, "model4_pred_results.npz"),
                file_pred_results=file_pred_results,
                total_pred_traj=total_pred_traj,
                total_true_traj=total_true_traj,
                loss_history=loss_history)
        print(f"\n✅ 所有预测结果已保存至: {save_dir}")
    except Exception as e:
        print(f"按文件ID预测出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Windows下修复多进程问题
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    
    run_training_and_viz()