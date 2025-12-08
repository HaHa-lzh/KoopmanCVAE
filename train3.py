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

# ===================== 模型导入 =====================
##注意！ model3 用于导入训练、预测函数及损失函数
##核心模型KoopmanCVAE
 ##前馈forward_single_pair
 #输入：状态对【初始时刻状态（归一化）x_i、末时刻状态（归一化）x_i_plus_1】【时间间隔（非归一化)delta_t】 
 #形成标签：末时刻状态（归一化）的标签是初始时刻状态（归一化）、时间间隔（非归一化)用于潜在空间标记
 #输出：psi_i, psi_i_plus_1, K_matrix, psi_recon, x_i_recon, x_i_plus_1_recon, mu, log_var
 ##预测函数predict_long_term：用标签计算对应末状态
##损失函数loss_cvae_koopman
 # 1. psi_recon_loss: K*psi_i ≈ psi_{i+1}（核心单步重构）
 # 2. x_recon_loss: X重构损失
 # 3. kl_loss: CVAE潜在分布KL散度
 # 4. k_product_loss: 多步K乘积误差（仅K_product不为None时生效）——————注意！本代码中没有使用这部分（计算量太大了）
 # 5. latent_consistency_loss: 按文件分组约束mu/log_var（同文件内样本接近）
from model3 import KoopmanCVAE, loss_cvae_koopman

##注意！ dataset3 用于导入训练、预测数据
## load_orbital_file：return 
##训练
# 所有相邻状态对（状态归一化、时间非归一化）all_adj_pairs, 
# 所有非相邻状态对（状态归一化、时间非归一化）all_non_adj_pairs, 
# 归一化参数scaler, 
# 训练文件编号:file_mapping
##测试
# 测试集的非归一化状态full_data, 
# 测试集的非归一化时间full_timestamps, 
from dataset3 import load_orbital_file  # 已修改为返回file_mapping

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ===================== Collate函数（支持file_id） =====================
def collate_fn_unified(batch):
    """统一collate函数 新增file_id传递"""
    Xi_list = []
    Xj_list = []
    delta_t_list = []
    file_id_list = []  # 新增：存储每个样本的file_id
    
    for item in batch:
        # 兼容格式：(Xi,Xj,delta_t,mid_indices,file_id) 或旧格式
        if len(item) == 3:
            Xi, Xj, delta_t = item
            file_id = 0  # 旧格式默认file_id=0
        elif len(item) == 4:
            Xi, Xj, delta_t, _ = item  # 忽略mid_indices
            file_id = 0
        elif len(item) == 5:  # 新增：适配带file_id的格式
            Xi, Xj, delta_t, _, file_id = item
        else:
            continue
        
        # 确保所有tensor都是2D (1, 6) / (1, 1)
        if Xi.dim() == 1:
            Xi = Xi.unsqueeze(0)  # (6,) -> (1, 6)
        if Xj.dim() == 1:
            Xj = Xj.unsqueeze(0)  # (6,) -> (1, 6)
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)  # (1,) -> (1, 1)
        
        Xi_list.append(Xi)
        Xj_list.append(Xj)
        delta_t_list.append(delta_t)
        file_id_list.append(torch.tensor([file_id], dtype=torch.long))  # 新增：转为tensor
    
    # 堆叠为3D batch (B, 1, 6) / (B, 1, 1)
    Xi_batch = torch.cat(Xi_list, dim=0)
    Xj_batch = torch.cat(Xj_list, dim=0)
    delta_t_batch = torch.cat(delta_t_list, dim=0)
    file_id_batch = torch.cat(file_id_list, dim=0)  # 新增：(B, 1)
    
    return Xi_batch, Xj_batch, delta_t_batch, file_id_batch  # 新增：返回file_id

# ===================== Dataset类（支持file_id） =====================
class StatePairDataset(Dataset):
    """统一的状态对Dataset类（新增file_id读取）"""
    def __init__(self, state_pairs):
        self.state_pairs = state_pairs
        
    def __len__(self):
        return len(self.state_pairs)
    
    def __getitem__(self, idx):
        try:
            pair = self.state_pairs[idx]
            # 兼容格式：(Xi,Xj,delta_t) / (Xi,Xj,delta_t,mid_indices) / (Xi,Xj,delta_t,mid_indices,file_id)
            if len(pair) == 3:
                Xi, Xj, delta_t = pair
                file_id = 0
            elif len(pair) == 4:
                Xi, Xj, delta_t, _ = pair  # 丢弃mid_indices
                file_id = 0
            elif len(pair) == 5:  # 新增：适配带file_id的格式
                Xi, Xj, delta_t, _, file_id = pair
            else:
                raise ValueError(f"无效的状态对格式，长度：{len(pair)}")
            
            # 统一转换为tensor（基础维度：1D）
            Xi = torch.FloatTensor(Xi) if not isinstance(Xi, torch.Tensor) else Xi
            Xj = torch.FloatTensor(Xj) if not isinstance(Xj, torch.Tensor) else Xj
            delta_t = torch.FloatTensor([delta_t])  # 1D (1,)
            file_id = torch.tensor(file_id, dtype=torch.long)  # 新增：file_id转为long型tensor
            
            return Xi, Xj, delta_t, file_id  # 新增：返回file_id
        except Exception as e:
            print(f"处理索引{idx}出错: {e}")
            # 返回默认值（1D）
            default = torch.zeros(6, dtype=torch.float32)
            return default, default, torch.zeros(1, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

# ===================== 封装预测+绘图+保存函数=====================
def predict_and_plot(model, scaler, full_data, full_timestamps, pred_steps, device, save_dir, epoch_num):
    """
    封装长时预测+绘图+保存逻辑，便于每100Epoch调用
    Args:
        model: 训练中的KoopmanCVAE模型
        scaler: 数据归一化器
        full_data: 完整轨道数据
        full_timestamps: 完整时间戳
        pred_steps: 预测步数
        device: 计算设备（cuda/cpu）
        save_dir: 保存目录
        epoch_num: 当前Epoch编号（用于文件名）
    """
    print(f"\n=== Epoch {epoch_num} 阶段性预测与绘图 ===")
    # 1. 准备初始点
    test_start_idx = 0
    full_data = full_data.to(device) if isinstance(full_data, torch.Tensor) else torch.FloatTensor(full_data).to(device)
    x0_norm = full_data[test_start_idx:test_start_idx+1]  # (1, 6)
    
    # 2. 构建Δt序列
    deltat_seq = []
    if full_timestamps is not None and len(full_timestamps) >= test_start_idx + pred_steps:
        for i in range(test_start_idx, test_start_idx + pred_steps):
            if i+1 < len(full_timestamps):
                dt = full_timestamps[i+1].item() - full_timestamps[i].item()
            else:
                dt = full_timestamps[i].item() - full_timestamps[i-1].item()
            deltat_seq.append(dt)
    else:
        # 从所有状态对中计算平均Δt（这里简化：用固定值，实际可传all_pairs）
        dt_mean = 1.0
        deltat_seq = [dt_mean] * pred_steps
    
    # 3. Δt序列处理（设备/归一化）
    deltat_seq = torch.FloatTensor(deltat_seq).unsqueeze(1).to(device)

    ####注意！！！！！这里没有用deltat_seq_norm，训练和预测中的时间都是非归一化的deltat
    deltat_seq_norm = deltat_seq
    if scaler and hasattr(scaler, 'dt_mean') and scaler.dt_mean is not None:
        scaler.dt_mean = scaler.dt_mean.to(device) if isinstance(scaler.dt_mean, torch.Tensor) else torch.tensor(scaler.dt_mean, device=device)
        scaler.dt_std = scaler.dt_std.to(device) if isinstance(scaler.dt_std, torch.Tensor) else torch.tensor(scaler.dt_std, device=device)
        deltat_seq_norm = scaler.transform_dt(deltat_seq)
    
    # 4. 长时预测
    model.eval()
    with torch.no_grad():
        pred_traj_norm, K_matrices = model.predict_long_term(
            x0=x0_norm,
            delta_t_seq=deltat_seq,
            steps=pred_steps,
            use_prior=True
        )
    
    # 5. 反归一化（对状态量）
    if isinstance(pred_traj_norm, np.ndarray):
        pred_traj_norm_tensor = torch.tensor(pred_traj_norm, dtype=torch.float32).to(device)
    elif isinstance(pred_traj_norm, list):
        pred_traj_norm_tensor = torch.FloatTensor(pred_traj_norm).to(device)
    else:
        pred_traj_norm_tensor = pred_traj_norm.to(device)
    pred_traj_real = scaler.inverse_transform(pred_traj_norm_tensor).cpu().numpy() if scaler else pred_traj_norm_tensor.cpu().numpy()
    
    # 6. 真实轨迹
    true_traj_norm = full_data[test_start_idx : test_start_idx + pred_steps + 1]
    true_traj_real = scaler.inverse_transform(true_traj_norm).cpu().numpy() if scaler else true_traj_norm.cpu().numpy()
    
    # 7. 构建时间轴
    deltat_seq_real = deltat_seq.cpu().numpy()
    deltat_seq_real_flat = deltat_seq_real.flatten()
    cumulative_time = np.cumsum(deltat_seq_real_flat)
    time_axis = np.insert(cumulative_time, 0, 0.0)
    total_time = cumulative_time[-1] if len(cumulative_time) > 0 else 0.0
    
    # 8. 绘图
    fig = plt.figure(figsize=(22, 14))

    # Loss曲线（阶段性绘图时Loss曲线只显示到当前Epoch）
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
    ax2.set_title(f"Orbital Prediction ({pred_steps} steps, Epoch {epoch_num})")
    ax2.legend()

    # 6个状态量对比
    state_names = [
        'X Position (km)', 'Y Position (km)', 'Z Position (km)',
        'Vx Velocity (km/s)', 'Vy Velocity (km/s)', 'Vz Velocity (km/s)'
    ]
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

    for i in range(6):
        ax = fig.add_subplot(3, 3, i+4)
        ax.plot(time_axis, true_traj_real[:, i], color=colors[i], 
                linestyle='-', lw=1.5, alpha=0.8, label='True Value')
        ax.plot(time_axis, pred_traj_real[:, i], color=colors[i], 
                linestyle='--', lw=1.5, alpha=0.8, label='Pred Value')
        ax.set_title(f"{state_names[i]} (Epoch {epoch_num})")
        ax.set_xlabel(f"Time (s) (Total: {total_time:.1f}s)")
        ax.set_ylabel(state_names[i].split(' ')[0])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 9. 保存阶段性结果
    plot_filename = f"unified_prediction_result_{epoch_num}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Epoch {epoch_num} 预测图已保存至: {plot_path}")
    
    # 关闭画布释放内存
    plt.close(fig)

def run_training_and_viz():
    # 文件名配置 #####注意！！！！这里的训练集为data文件夹中的5个轨道
    data_dir = "data" 
    if os.path.exists(data_dir):
        FILENAME = data_dir
    else:
        filename_base = "轨道1-周期90.0min-半长轴6652.6km-e0.000-倾角90.0°-升交点0.0°-近地点0.0°.txt"
        possible_paths = [
            os.path.join("data", filename_base),
            filename_base
        ]
        FILENAME = None
        for path in possible_paths:
            if os.path.exists(path):
                FILENAME = path
                break
            
    if FILENAME is None:
        print(f"错误: 找不到数据文件或目录")
        return

    # 参数配置 
    # #####注意！！！！这里为训练超参数；NON_ADJ_RATIO指的是每个文件形成的状态对数量
    STATE_DIM = 6
    OBS_DIM = 64
    K_LATENT_DIM = 32
    BATCH_SIZE = 64
    EPOCHS = 1000
    LR = 1e-4
    LAMBDA_K_PRODUCT = 10.0
    NON_ADJ_RATIO = 2 # 保留参数但实际不区分
    PRED_STEPS = 5000  # 预测步数
    PLOT_INTERVAL = 100  # 每100Epoch绘图一次
    DELTA = 1.0  # 新增：分组一致性损失权重

    # 1. 数据准备
    print("加载轨道数据并生成状态对...")
    
    adj_pairs, non_adj_pairs, scaler, full_data, full_timestamps, file_mapping = load_orbital_file(
        FILENAME, non_adj_ratio=NON_ADJ_RATIO
    )
    
    # 合并所有状态对（相邻+非相邻）####注意！！这里没有将相邻状态对纳入训练集中
    all_pairs = []
    #if adj_pairs and len(adj_pairs) > 0:
        #all_pairs.extend(adj_pairs)
    if non_adj_pairs and len(non_adj_pairs) > 0:
        all_pairs.extend(non_adj_pairs)
    
    if len(all_pairs) == 0:
        print("无法继续：无有效状态对。")
        return

    # 构建统一的Dataset和DataLoader（支持file_id）
    unified_dataset = StatePairDataset(all_pairs)
    unified_dataloader = DataLoader(
        unified_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn_unified,  # 使用扩展后的collate函数
        drop_last=True
    )
    
    print(f"训练数据统计：")
    print(f"  总状态对数量: {len(unified_dataset)}")
    print(f"  其中相邻对数量: {len(adj_pairs) if adj_pairs else 0}")
    print(f"  其中非相邻对数量: {len(non_adj_pairs) if non_adj_pairs else 0}")
    print(f"  文件编号映射: {file_mapping}")  # 新增：打印文件映射关系
    
    # 2. 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = KoopmanCVAE(STATE_DIM, OBS_DIM, K_LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. 训练循环
    print("开始训练...")
    global loss_history  # 声明为全局变量，便于绘图函数访问
    loss_history = []
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)  # 提前创建保存目录
    
    model.train()
    for epoch in range(EPOCHS):
        total_epoch_loss = 0
        total_epoch_psi_loss =0
        total_epoch_x_loss =0
        total_epoch_kl_loss =0
        total_epoch_latent_loss =0
        batch_num = 0
        
        # ---------------------- 统一训练所有状态对 ----------------------
        progress = tqdm(unified_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in progress:
            try:
                # batch新增file_id_batch，维度：(B, 1, 6), (B, 1, 6), (B, 1, 1), (B, 1)
                Xi_batch, Xj_batch, delta_t_batch, file_id_batch = [b.to(device) for b in batch]
                
                # 压缩维度：(B, 1, 6) -> (B, 6)；(B, 1, 1) -> (B, 1)；(B, 1) -> (B,)
                Xi_batch = Xi_batch.squeeze(1)
                Xj_batch = Xj_batch.squeeze(1)
                delta_t_batch = delta_t_batch.squeeze(1)
                #file_id_batch = file_id_batch.squeeze(1)  # 新增：压缩为(B,)
                
                optimizer.zero_grad()
                
                # 前向计算（统一按相邻对逻辑）
                psi_i, psi_j, K_matrix, psi_recon, x_i_recon, x_j_recon, mu, log_var = model.forward_single_pair(
                    Xi_batch, Xj_batch, delta_t_batch
                )
                
                # 修改6：调用带file_id的loss函数，新增latent_consistency_loss返回值
                loss, psi_loss, x_loss, kl_loss, k_product_loss, latent_loss = loss_cvae_koopman(
                    psi_recon, psi_j,
                    x_i_recon, Xi_batch,
                    x_j_recon, Xj_batch,
                    mu, log_var, K_matrix,
                    file_ids=file_id_batch,  # 传入file_id
                    lambda_k_product=LAMBDA_K_PRODUCT,
                    delta=DELTA  # 新增：分组一致性损失权重
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
                total_epoch_latent_loss +=latent_loss.item()
                batch_num += 1
                
                # 修改7：进度条新增latent_loss显示
                progress.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    #'latent_loss': f"{latent_loss.item():.6f}"
                })
            except Exception as e:
                print(f"\n批次处理出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ---------------------- 计算平均损失 ----------------------
        avg_loss = total_epoch_loss / batch_num if batch_num > 0 else 0
        avg_psi_loss = total_epoch_psi_loss / batch_num if batch_num > 0 else 0
        avg_x_loss = total_epoch_x_loss / batch_num if batch_num > 0 else 0
        avg_kl_loss = total_epoch_kl_loss / batch_num if batch_num > 0 else 0
        avg_latent_loss = total_epoch_latent_loss / batch_num if batch_num > 0 else 0

        loss_history.append(avg_loss)
        
        # 打印日志
        print(f"\nEpoch {epoch + 1}: ")
        print(f"  Average Loss = {avg_loss:.6f}(Average psi_loss = {avg_psi_loss:.6f}, Average x_loss = {avg_x_loss:.6f},Average kl_loss = {avg_kl_loss:.6f},Average latent_loss = {avg_latent_loss:.6f})")

        # ---------------------- 每100Epoch绘制并保存预测图 ----------------------
        current_epoch = epoch + 1  # 转为1-based编号
        if current_epoch % PLOT_INTERVAL == 0:
            predict_and_plot(
                model=model,
                scaler=scaler,
                full_data=full_data,
                full_timestamps=full_timestamps,
                pred_steps=PRED_STEPS,
                device=device,
                save_dir=save_dir,
                epoch_num=current_epoch
            )
            model.train()  # 绘图后切回训练模式



    # 4. 最终长时程预测+绘图+保存（基本逻辑和封装的绘制一样！）
    print("\n=== 最终训练完成，执行最终预测与保存 ===")
    test_start_idx = 0
    
    # 准备初始点（统一为2D）
    full_data = full_data.to(device) if isinstance(full_data, torch.Tensor) else torch.FloatTensor(full_data).to(device)
    x0_norm = full_data[test_start_idx:test_start_idx+1]  # (1, 6)
    
    # 构建Δt序列（统一为2D + 确保设备匹配）
    deltat_seq = []
    if full_timestamps is not None and len(full_timestamps) >= test_start_idx + PRED_STEPS:
        for i in range(test_start_idx, test_start_idx + PRED_STEPS):
            if i+1 < len(full_timestamps):
                dt = full_timestamps[i+1].item() - full_timestamps[i].item()
            else:
                dt = full_timestamps[i].item() - full_timestamps[i-1].item()
            deltat_seq.append(dt)
    else:
        # 从所有状态对中计算平均Δt
        dt_list = [pair[2] for pair in all_pairs[:100] if len(pair)>=3]
        dt_mean = np.mean(dt_list) if dt_list else 1.0
        deltat_seq = [dt_mean] * PRED_STEPS
    deltat_seq_print = deltat_seq[:100]
    # 打印基础信息
    print(f"\n=== 最终预测 - 前100个Δt序列信息 ===")
    print(f"Δt序列总长度: {len(deltat_seq)}")
    print(f"打印前{len(deltat_seq_print)}个Δt值:")
    for i in range(0, len(deltat_seq_print), 10):
        end_idx = min(i+10, len(deltat_seq_print))
        dt_line = [f"{dt:.6f}" for dt in deltat_seq_print[i:end_idx]]
        print(f"第{i+1}-{end_idx}个Δt: {', '.join(dt_line)}")
    
    # Δt序列创建时直接指定设备
    deltat_seq = torch.FloatTensor(deltat_seq).unsqueeze(1).to(device)
    
    # 归一化前确保scaler参数在同一设备
    ####注意！！！！！这里没有用deltat_seq_norm，训练和预测中的时间都是非归一化的deltat
    deltat_seq_norm = deltat_seq  # 默认值
    if scaler and hasattr(scaler, 'dt_mean') and scaler.dt_mean is not None:
        scaler.dt_mean = scaler.dt_mean.to(device) if isinstance(scaler.dt_mean, torch.Tensor) else torch.tensor(scaler.dt_mean, device=device)
        scaler.dt_std = scaler.dt_std.to(device) if isinstance(scaler.dt_std, torch.Tensor) else torch.tensor(scaler.dt_std, device=device)
        deltat_seq_norm = scaler.transform_dt(deltat_seq)
    
    # 预测（此时所有输入都在CUDA）
    pred_traj_norm, K_matrices = model.predict_long_term(
        x0=x0_norm,
        delta_t_seq=deltat_seq,
        steps=PRED_STEPS,
        use_prior=True
    )
    
    # 反归一化（确保维度正确）
    if isinstance(pred_traj_norm, np.ndarray):
        pred_traj_norm_tensor = torch.tensor(pred_traj_norm, dtype=torch.float32).to(device)
    elif isinstance(pred_traj_norm, list):
        pred_traj_norm_tensor = torch.FloatTensor(pred_traj_norm).to(device)
    else:
        pred_traj_norm_tensor = pred_traj_norm.to(device)
    
    pred_traj_real = scaler.inverse_transform(pred_traj_norm_tensor).cpu().numpy() if scaler else pred_traj_norm_tensor.cpu().numpy()
    
    # 真实轨迹（统一维度）
    true_traj_norm = full_data[test_start_idx : test_start_idx + PRED_STEPS + 1]
    true_traj_real = scaler.inverse_transform(true_traj_norm).cpu().numpy() if scaler else true_traj_norm.cpu().numpy()

    # 构建时间轴
    deltat_seq_real = deltat_seq.cpu().numpy()
    deltat_seq_real_flat = deltat_seq_real.flatten()
    deltat_seq_print = deltat_seq_real_flat[:100]
    print(f"\n=== 最终预测 - deltat_seq_real 前100个值（真实Δt，单位：秒）===")
    print(f"deltat_seq_real 总长度: {len(deltat_seq_real_flat)}")
    print(f"打印前{len(deltat_seq_print)}个真实Δt值:")
    for i in range(0, len(deltat_seq_print), 10):
        end_idx = min(i + 10, len(deltat_seq_print))
        dt_line = [f"{dt:.6f}" for dt in deltat_seq_print[i:end_idx]]
        print(f"第{i+1}-{end_idx}个Δt: {', '.join(dt_line)}")
    
    cumulative_time = np.cumsum(deltat_seq_real_flat)
    time_axis = np.insert(cumulative_time, 0, 0.0)
    total_time = cumulative_time[-1] if len(cumulative_time) > 0 else 0.0
    print(f"\n最终预测{PRED_STEPS}步的累计总时间: {total_time:.6f} 秒")

    # 5. 最终绘图
    fig = plt.figure(figsize=(22, 14))

    # Loss曲线
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(loss_history, color='navy', label='Total Loss')
    ax1.set_title("Training Loss (Final)")
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
    ax2.set_title(f"Orbital Prediction ({PRED_STEPS} steps, Final)")
    ax2.legend()

    # 6个状态量对比
    state_names = [
        'X Position (km)', 'Y Position (km)', 'Z Position (km)',
        'Vx Velocity (km/s)', 'Vy Velocity (km/s)', 'Vz Velocity (km/s)'
    ]
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

    for i in range(6):
        ax = fig.add_subplot(3, 3, i+4)
        ax.plot(time_axis, true_traj_real[:, i], color=colors[i], 
                linestyle='-', lw=1.5, alpha=0.8, label='True Value')
        ax.plot(time_axis, pred_traj_real[:, i], color=colors[i], 
                linestyle='--', lw=1.5, alpha=0.8, label='Pred Value')
        ax.set_title(state_names[i])
        ax.set_xlabel(f"Time (s) (Total: {total_time:.1f}s)")
        ax.set_ylabel(state_names[i].split(' ')[0])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 6. 保存最终结果
    # 保存模型
    model_path = os.path.join(save_dir, "koopman_cvae_unified_model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n最终模型已保存至: {model_path}")
    
    # 保存最终可视化
    plot_path = os.path.join(save_dir, "unified_prediction_result_final.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"最终可视化结果已保存至: {plot_path}")
    
    # 保存训练数据
    data_path = os.path.join(save_dir, "unified_training_results_final.npz")
    np.savez(data_path, 
             loss_history=np.array(loss_history),
             pred_traj=pred_traj_real,
             true_traj=true_traj_real,
             time_axis=time_axis,
             total_time=total_time)
    print(f"最终训练数据已保存至: {data_path}")
    
    # plt.show()

if __name__ == "__main__":
    run_training_and_viz()