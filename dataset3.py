import torch
import pandas as pd
import numpy as np
import os
import glob
import random

# ===================== 全局维度规范（适配model1） =====================
# 状态量维度：STATE_DIM = 6
# 时间戳维度：(N, 1) （2D）
# 单个状态维度：(STATE_DIM,) （1D）→ 输入model1时扩展为(1, STATE_DIM)
# 单个Δt维度：(1,) （1D tensor）→ 输入model1时扩展为(1, 1)
# 状态对中Xi/Xj维度：(STATE_DIM,) （1D）
# 状态对中delta_t维度：(1,) （1D tensor）→ 关键适配model1的拼接要求
# 状态对新增字段：file_id（int）- 所属文件编号
# =====================================================================

# 设置随机种子保证可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ===================== 新增：固定可视化文件路径配置 =====================
VISUALIZATION_FILE_PATH = r"C:\Users\admin\Desktop\深度学习\深度学习大作业\原始数据\轨道积分结果（仅二体）\轨道5-周期96.2min-半长轴6952.6km-e0.000-倾角90.0°-升交点30.0°-近地点0.0°.txt"

# ==========================================
# 数据归一化工具类（完全适配model1）
# ==========================================
class SimpleScaler:
    """
    用于将大幅度的轨道数据(如 6000km) 归一化到神经网络喜欢的 [-3, 3] 范围
    支持状态量和时间间隔Δt的独立归一化
    维度规范：
    - 输入/输出均为2D tensor: (N, dim)
    - 为model1提供1D→2D的便捷转换
    - 自动适配设备（CPU/CUDA）
    """
    def __init__(self):
        self.mean = None          # 2D: (1, state_dim)
        self.std = None           # 2D: (1, state_dim)
        self.dt_mean = None       # 2D: (1, 1)
        self.dt_std = None        # 2D: (1, 1)
        self.device = None        # 新增：记录设备
    
    def fit(self, data, dt_data=None):
        """
        拟合归一化参数
        Args:
            data: torch tensor (N, state_dim) 2D 状态量数据
            dt_data: torch tensor (N, 1) 2D Δt数据
        """
        # 记录数据所在设备
        self.device = data.device if isinstance(data, torch.Tensor) else torch.device('cpu')
        
        # 强制转换为2D（防止传入1D数据）
        if data.dim() == 1:
            data = data.unsqueeze(0)
        self.mean = data.mean(dim=0, keepdim=True).to(self.device)  # 固定到设备
        self.std = data.std(dim=0, keepdim=True).to(self.device) + 1e-6
        
        # 拟合Δt的归一化参数（强制2D）
        if dt_data is not None:
            if dt_data.dim() == 1:
                dt_data = dt_data.unsqueeze(1)
            self.dt_mean = dt_data.mean(dim=0, keepdim=True).to(self.device)  # 固定到设备
            self.dt_std = dt_data.std(dim=0, keepdim=True).to(self.device) + 1e-6

    def transform(self, data):
        """归一化状态量（输入2D/1D，输出2D）"""
        if self.mean is None:
            raise ValueError("Scaler not fitted yet!")
        # 强制转换为2D + 匹配设备
        if data.dim() == 1:
            data = data.unsqueeze(0)
        data = data.to(self.device)  # 新增：统一设备
        # 确保归一化参数在同一设备
        self.mean = self.mean.to(data.device)
        self.std = self.std.to(data.device)
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        """反归一化状态量（输入2D/1D，输出匹配输入维度）"""
        if self.mean is None:
            raise ValueError("Scaler not fitted yet!")
        input_dim = data.dim()
        # 强制转换为2D + 匹配设备
        if data.dim() == 1:
            data = data.unsqueeze(0)
        data = data.to(self.device)  # 新增：统一设备
        # 确保归一化参数在同一设备
        self.mean = self.mean.to(data.device)
        self.std = self.std.to(data.device)
        res = data * self.std + self.mean
        # 还原输入维度
        if input_dim == 1:
            return res.squeeze(0)
        return res
    
    # Δt 归一化/反归一化方法（专为model1优化 + 设备适配）
    def transform_dt(self, dt_data):
        """
        归一化Δt（支持1D tensor输入，输出2D）
        Args:
            dt_data: 1D tensor (B,) / 2D tensor (B,1)
        Returns:
            2D tensor (B,1)
        """
        if self.dt_mean is None:
            raise ValueError("DT Scaler not fitted yet!")
        # 强制转换为2D + 匹配设备
        if dt_data.dim() == 1:
            dt_data = dt_data.unsqueeze(1)
        dt_data = dt_data.to(self.device)  # 新增：统一设备
        # 确保归一化参数在同一设备
        self.dt_mean = self.dt_mean.to(dt_data.device)
        self.dt_std = self.dt_std.to(dt_data.device)
        return (dt_data - self.dt_mean) / self.dt_std
    
    def inverse_transform_dt(self, dt_data):
        """反归一化Δt（支持1D/2D输入，输出匹配输入维度）"""
        if self.dt_mean is None:
            raise ValueError("DT Scaler not fitted yet!")
        input_dim = dt_data.dim()
        # 强制转换为2D + 匹配设备
        if dt_data.dim() == 1:
            dt_data = dt_data.unsqueeze(1)
        dt_data = dt_data.to(self.device)  # 新增：统一设备
        # 确保归一化参数在同一设备
        self.dt_mean = self.dt_mean.to(dt_data.device)
        self.dt_std = self.dt_std.to(dt_data.device)
        res = dt_data * self.dt_std + self.dt_mean
        # 还原输入维度
        if input_dim == 1:
            return res.squeeze(1)
        return res

# ==========================================
# 核心函数：生成状态对（新增file_id记录）
# ==========================================
def generate_state_pairs(full_states, full_timestamps, file_id, non_adj_ratio=0):
    """
    从单个文件的完整轨迹生成状态对（严格匹配model1输入维度 + 记录file_id）
    Args:
        full_states: torch.Tensor (N, state_dim) 2D 完整状态序列（归一化后）
        full_timestamps: torch.Tensor (N, 1) 2D 完整时间戳序列（原始值）
        file_id: int 当前文件的唯一编号（如0,1,2...）
        non_adj_ratio: 随机非相邻状态对的数量比例（相对于相邻对数量）
    Returns:
        adj_pairs: list 相邻状态对，每个元素=(Xi, Xj, delta_t, mid_indices, file_id)
        non_adj_pairs: list 随机非相邻状态对，每个元素=(Xi, Xj, delta_t, mid_indices, file_id)
            - Xi/Xj: torch.Tensor (state_dim,) 1D 状态量
            - delta_t: torch.Tensor (1,) 1D tensor（关键修改：替代float）
            - mid_indices: list 中间索引
            - file_id: int 所属文件编号
    """
    # 维度校验（强制输入为2D）
    if full_states.dim() != 2:
        raise ValueError(f"full_states必须是2D tensor，当前维度：{full_states.dim()}")
    if full_timestamps.dim() != 2:
        raise ValueError(f"full_timestamps必须是2D tensor，当前维度：{full_timestamps.dim()}")
    
    n_total = len(full_states)
    adj_pairs = []
    non_adj_pairs = []
    
    # 1. 生成所有相邻状态对（Xi, Xi+1）
    for i in range(n_total - 1):
        # 提取状态（2D→1D，方便后续批次处理）
        Xi = full_states[i].squeeze(0)  # (1, 6) → (6,)
        Xj = full_states[i+1].squeeze(0)  # (1, 6) → (6,)
        # 提取时间戳并计算Δt（转为1D tensor）
        ti = full_timestamps[i].item()
        tj = full_timestamps[i+1].item()
        delta_t = torch.FloatTensor([tj - ti])  # 关键：1D tensor (1,) 替代float
        # 中间索引（相邻对只有i）
        mid_indices = [i]
        # 严格维度校验（适配model1）
        assert Xi.dim() == 1 and Xi.shape[0] == 6, f"Xi维度错误：{Xi.shape}"
        assert Xj.dim() == 1 and Xj.shape[0] == 6, f"Xj维度错误：{Xj.shape}"
        assert delta_t.dim() == 1 and delta_t.shape[0] == 1, f"delta_t维度错误：{delta_t.shape}"
        # 新增：附加file_id
        adj_pairs.append((Xi, Xj, delta_t, mid_indices, file_id))
    
    # 2. 生成随机非相邻状态对（Xi, Xj，j > i+1）
    n_adj = len(adj_pairs)
    n_non_adj = max(1, int(n_adj * non_adj_ratio))  # 至少生成1个
    
    for _ in range(n_non_adj):
        # 随机选择索引（保证j > i+1）
        i = random.randint(0, n_total - 3)
        j = random.randint(i + 2, n_total - 1)
        
        # 提取状态（2D→1D）
        Xi = full_states[i].squeeze(0)
        Xj = full_states[j].squeeze(0)
        # 提取时间戳并计算Δt（1D tensor）
        ti = full_timestamps[i].item()
        tj = full_timestamps[j].item()
        delta_t = torch.FloatTensor([tj - ti])  # 1D tensor (1,)
        # 中间索引
        mid_indices = list(range(i, j))
        # 严格维度校验
        assert Xi.dim() == 1 and Xi.shape[0] == 6, f"Xi维度错误：{Xi.shape}"
        assert Xj.dim() == 1 and Xj.shape[0] == 6, f"Xj维度错误：{Xj.shape}"
        assert delta_t.dim() == 1 and delta_t.shape[0] == 1, f"delta_t维度错误：{delta_t.shape}"
        # 新增：附加file_id
        non_adj_pairs.append((Xi, Xj, delta_t, mid_indices, file_id))
    
    print(f"文件{file_id}生成状态对统计：相邻对={len(adj_pairs)}, 非相邻对={len(non_adj_pairs)}")
    return adj_pairs, non_adj_pairs

# ==========================================
# 新增：单独加载可视化用的完整数据文件
# ==========================================
def load_visualization_data(file_path, scaler, state_dim=6):
    """
    加载指定的可视化用轨道文件，返回归一化后的完整状态和原始时间戳
    Args:
        file_path: str 可视化文件路径
        scaler: SimpleScaler 已拟合的归一化器
        state_dim: int 状态量维度（固定为6）
    Returns:
        norm_states: torch.Tensor (N, state_dim) 2D 归一化后的完整状态
        raw_timestamps: torch.Tensor (N, 1) 2D 原始时间戳
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"可视化文件不存在：{file_path}")
    
    print(f"\n加载可视化专用文件：{file_path}")
    # 读取文件（兼容不同分隔符）
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None)
    except:
        df = pd.read_csv(file_path, sep='\t', header=None)
    
    # 提取数据并强制转换为2D
    # 时间戳：(N,) → (N, 1)
    timestamps = df.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
    # 状态量：(N, state_dim) （强制截取前6列）
    states = df.iloc[:, 1:1+state_dim].values.astype(np.float32)
    
    # 转换为tensor（确保2D）
    raw_ts_tensor = torch.FloatTensor(timestamps)  # (N, 1) 2D
    raw_states_tensor = torch.FloatTensor(states)  # (N, 6) 2D
    
    # 维度校验
    assert raw_ts_tensor.dim() == 2 and raw_ts_tensor.shape[1] == 1, \
        f"可视化文件时间戳维度错误：{raw_ts_tensor.shape}（要求(N,1)）"
    assert raw_states_tensor.dim() == 2 and raw_states_tensor.shape[1] == state_dim, \
        f"可视化文件状态量维度错误：{raw_states_tensor.shape}（要求(N,{state_dim})）"
    
    # 使用全局归一化器归一化状态量（输入2D，输出2D）
    norm_states = scaler.transform(raw_states_tensor)
    
    print(f"可视化文件加载完成：状态量维度={norm_states.shape}，时间戳维度={raw_ts_tensor.shape}")
    return norm_states, raw_ts_tensor

# ==========================================
# 重构数据加载函数（新增file_id记录 + 指定可视化文件）
# ==========================================
def load_orbital_file(path, non_adj_ratio=0.2, state_dim=6):
    """
    加载轨道文件并生成状态对（输出完全匹配model1输入要求 + 记录file_id + 指定可视化文件）
    Args:
        path: str 文件/目录路径（训练数据）
        non_adj_ratio: float 非相邻状态对比例
        state_dim: int 状态量维度（固定为6）
    Returns:
        all_adj_pairs: list 所有相邻状态对
            - Xi/Xj: (6,) 1D tensor
            - delta_t: (1,) 1D tensor
            - mid_indices: list
            - file_id: int 所属文件编号
        all_non_adj_pairs: list 所有非相邻状态对（同左）
        scaler: SimpleScaler 归一化器（适配model1的维度转换）
        full_data: torch.Tensor (N, state_dim) 2D 指定可视化文件的归一化状态
        full_timestamps: torch.Tensor (N, 1) 2D 指定可视化文件的原始时间戳
        file_mapping: dict 文件编号到文件名的映射（如{0: "轨道1.txt", 1: "轨道2.txt"}）
    """
    # 路径处理（训练数据）
    if os.path.isdir(path):
        print(f"从目录读取训练数据: {path}")
        file_paths = sorted(glob.glob(os.path.join(path, "*.txt")))
    else:
        print(f"读取单文件训练数据: {path}")
        file_paths = [path]
        
    if not file_paths:
        print("未找到训练数据文件")
        return None, None, None, None, None, None

    # 新增：文件编号到文件名的映射（训练数据）
    file_mapping = {}
    for idx, fp in enumerate(file_paths):
        file_mapping[idx] = os.path.basename(fp)
    
    # 存储所有训练文件的原始数据（严格2D）
    all_raw_states = []      # 每个元素：(N, state_dim) 2D
    all_raw_timestamps = []  # 每个元素：(N, 1) 2D
    all_raw_dt = []          # 每个元素：(N-1, 1) 2D
    
    # 1. 读取所有训练文件（强制2D）
    for fp in file_paths:
        try:
            # 兼容不同分隔符
            try:
                df = pd.read_csv(fp, sep='\s+', header=None)
            except:
                df = pd.read_csv(fp, sep='\t', header=None)
            
            # 提取数据并强制转换为2D
            # 时间戳：(N,) → (N, 1)
            timestamps = df.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
            # 状态量：(N, state_dim) （强制截取前6列）
            states = df.iloc[:, 1:1+state_dim].values.astype(np.float32)
            
            # 转换为tensor（确保2D）
            ts_tensor = torch.FloatTensor(timestamps)  # (N, 1) 2D
            states_tensor = torch.FloatTensor(states)  # (N, 6) 2D
            
            # 维度校验
            assert ts_tensor.dim() == 2 and ts_tensor.shape[1] == 1, \
                f"训练文件时间戳维度错误：{ts_tensor.shape}（要求(N,1)）"
            assert states_tensor.dim() == 2 and states_tensor.shape[1] == state_dim, \
                f"训练文件状态量维度错误：{states_tensor.shape}（要求(N,{state_dim})）"
            
            # 存储
            all_raw_states.append(states_tensor)
            all_raw_timestamps.append(ts_tensor)
            
            # 计算相邻Δt（2D）
            if len(timestamps) > 1:
                dt = np.diff(timestamps, axis=0)  # (N-1, 1)
                all_raw_dt.append(torch.FloatTensor(dt))
            else:
                all_raw_dt.append(torch.FloatTensor([[0.0]]))
                
        except Exception as e:
            print(f"读取训练文件 {fp} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    if not all_raw_states:
        print("无有效训练数据文件")
        return None, None, None, None, None, None
        
    print(f"成功读取 {len(all_raw_states)} 个训练文件")
    print(f"训练文件编号映射：{file_mapping}")
        
    # 2. 全局归一化（输入均为2D）
    # 拼接所有训练状态量（2D）
    all_states_concat = torch.cat(all_raw_states, dim=0)  # (total_N, 6)
    # 拼接所有训练Δt（2D）
    all_dt_concat = torch.cat(all_raw_dt, dim=0) if all_raw_dt else torch.FloatTensor([[0.0]])
    
    # 初始化并拟合归一化器
    scaler = SimpleScaler()
    scaler.fit(all_states_concat, dt_data=all_dt_concat)
    
    # 3. 为每个训练文件生成状态对（分配唯一file_id）
    all_adj_pairs = []
    all_non_adj_pairs = []
    
    for file_id, (raw_states, raw_ts) in enumerate(zip(all_raw_states, all_raw_timestamps)):
        # 归一化当前训练文件的状态量（输入2D，输出2D）
        norm_states = scaler.transform(raw_states)
        
        # 维度校验
        assert norm_states.dim() == 2 and norm_states.shape[1] == state_dim, \
            f"训练文件归一化状态维度错误：{norm_states.shape}"
        
        # 生成状态对（传递file_id，输出适配model1）
        adj_pairs, non_adj_pairs = generate_state_pairs(
            norm_states, raw_ts, file_id=file_id, non_adj_ratio=non_adj_ratio
        )
        
        # 合并到全局列表
        all_adj_pairs.extend(adj_pairs)
        all_non_adj_pairs.extend(non_adj_pairs)
    
    # 4. 加载指定的可视化文件（核心修改）
    try:
        full_data, full_timestamps = load_visualization_data(VISUALIZATION_FILE_PATH, scaler, state_dim)
    except Exception as e:
        print(f"加载可视化文件失败：{e}")
        # 降级方案：使用最后一个训练文件作为可视化数据
        print("降级为使用最后一个训练文件作为可视化数据")
        file_id = len(all_raw_states) - 1
        full_data = scaler.transform(all_raw_states[file_id])
        full_timestamps = all_raw_timestamps[file_id]
    
    # 5. 最终数据校验（确保输出完全匹配model1 + file_id）
    if not all_adj_pairs:
        print("未生成任何相邻状态对")
        return None, None, None, None, None, None
    
    # 随机打乱训练状态对（可选，提升训练效果）
    random.shuffle(all_adj_pairs)
    if all_non_adj_pairs:
        random.shuffle(all_non_adj_pairs)
    
    # 最终维度校验（抽样）
    sample_adj = all_adj_pairs[0]
    assert isinstance(sample_adj[0], torch.Tensor) and sample_adj[0].shape == (6,), \
        "相邻对Xi维度错误"
    assert isinstance(sample_adj[1], torch.Tensor) and sample_adj[1].shape == (6,), \
        "相邻对Xj维度错误"
    assert isinstance(sample_adj[2], torch.Tensor) and sample_adj[2].shape == (1,), \
        "相邻对delta_t维度错误"
    assert isinstance(sample_adj[4], int), "相邻对file_id必须是整数"
    
    print(f"最终状态对统计：总相邻对={len(all_adj_pairs)}, 总非相邻对={len(all_non_adj_pairs)}")
    print("✅ 数据加载完成，所有输出维度完全匹配model1要求（含file_id记录 + 指定可视化文件）！")
    return all_adj_pairs, all_non_adj_pairs, scaler, full_data, full_timestamps, file_mapping

# ===================== 适配训练的Dataset类（支持file_id） =====================
class StatePairDataset(torch.utils.data.Dataset):
    """
    支持file_id的状态对Dataset（适配model1 + 分组一致性损失）
    输出：(Xi, Xj, delta_t, file_id)
    """
    def __init__(self, state_pairs):
        self.state_pairs = state_pairs
        
    def __len__(self):
        return len(self.state_pairs)
    
    def __getitem__(self, idx):
        try:
            pair = self.state_pairs[idx]
            # 兼容新旧格式：(Xi,Xj,delta_t,mid_indices) 或 (Xi,Xj,delta_t,mid_indices,file_id)
            if len(pair) == 4:
                Xi, Xj, delta_t, _ = pair
                file_id = 0  # 旧格式默认file_id=0
            elif len(pair) == 5:
                Xi, Xj, delta_t, _, file_id = pair
            else:
                raise ValueError(f"无效的状态对格式，长度：{len(pair)}")
            
            # 统一转换为tensor（基础维度：1D）
            Xi = torch.FloatTensor(Xi) if not isinstance(Xi, torch.Tensor) else Xi
            Xj = torch.FloatTensor(Xj) if not isinstance(Xj, torch.Tensor) else Xj
            delta_t = torch.FloatTensor([delta_t]) if not isinstance(delta_t, torch.Tensor) else delta_t
            file_id = torch.tensor(file_id, dtype=torch.long)  # 转为long型tensor
            
            # 维度校验
            if Xi.dim() == 1:
                Xi = Xi.unsqueeze(0)  # (6,) → (1,6)
            if Xj.dim() == 1:
                Xj = Xj.unsqueeze(0)  # (6,) → (1,6)
            if delta_t.dim() == 1:
                delta_t = delta_t.unsqueeze(0)  # (1,) → (1,1)
            
            return Xi.squeeze(0), Xj.squeeze(0), delta_t.squeeze(0), file_id
        except Exception as e:
            print(f"处理索引{idx}出错: {e}")
            # 返回默认值
            default = torch.zeros(6, dtype=torch.float32)
            return default, default, torch.zeros(1, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

# ===================== 测试函数（验证file_id + model1兼容性） =====================
def test_model1_compatibility():
    """测试输出是否完全匹配model1的输入要求（含file_id）"""
    # 模拟model1的forward_single_pair输入逻辑
    def simulate_model_input(xi, xj, delta_t):
        """模拟model1的输入维度转换"""
        # 1D→2D（model1内部逻辑）
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)  # (6,) → (1,6)
        if xj.dim() == 1:
            xj = xj.unsqueeze(0)  # (6,) → (1,6)
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)  # (1,) → (1,1)
        
        # 模拟拼接操作（model1的核心逻辑）
        obs_dim = 64  # 模拟encoder输出维度
        psi_i = torch.randn(1, obs_dim)
        psi_j = torch.randn(1, obs_dim)
        cvae_input = torch.cat([psi_i, psi_j, delta_t], dim=1)
        
        assert cvae_input.dim() == 2, "拼接后必须是2D"
        assert cvae_input.shape == (1, obs_dim*2 + 1), \
            f"拼接维度错误：{cvae_input.shape}（要求(1, {obs_dim*2+1})）"
        return True
    
    # 生成测试数据
    test_states = torch.randn(100, 6)  # 2D (100,6)
    test_ts = torch.arange(100).float().unsqueeze(1)  # 2D (100,1)
    
    # 生成状态对（带file_id）
    adj_pairs, non_adj_pairs = generate_state_pairs(test_states, test_ts, file_id=0)
    
    # 测试相邻对（含file_id）
    Xi, Xj, delta_t, mid_indices, file_id = adj_pairs[0]
    assert file_id == 0, "file_id记录错误"
    assert simulate_model_input(Xi, Xj, delta_t), "相邻对不兼容model1"
    
    # 测试非相邻对（含file_id）
    Xi, Xj, delta_t, mid_indices, file_id = non_adj_pairs[0]
    assert file_id == 0, "file_id记录错误"
    assert simulate_model_input(Xi, Xj, delta_t), "非相邻对不兼容model1"
    
    # 测试Dataset
    dataset = StatePairDataset(adj_pairs)
    xi, xj, dt, fid = dataset[0]
    assert fid.item() == 0, "Dataset返回的file_id错误"
    assert xi.shape == (6,) and xj.shape == (6,) and dt.shape == (1,), "Dataset维度错误"
    
    print("✅ Model1兼容性测试通过（含file_id）！所有状态对均可直接输入model1！")

if __name__ == "__main__":
    # 运行维度规范测试
    test_model1_compatibility()
    
    # 测试数据加载（训练数据路径可自定义，可视化数据已固定）
    # 示例：训练数据从"data"目录读取，可视化数据使用固定路径
    adj_pairs, non_adj_pairs, scaler, full_data, full_ts, file_mapping = load_orbital_file("data")
    
    # 验证可视化数据
    if full_data is not None:
        print(f"\n✅ 可视化数据验证：")
        print(f"  可视化数据维度：{full_data.shape}")
        print(f"  可视化时间戳维度：{full_ts.shape}")
        print(f"  可视化数据归一化均值：{torch.mean(full_data).item():.6f}")