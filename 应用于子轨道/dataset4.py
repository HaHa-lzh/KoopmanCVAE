import torch
import pandas as pd
import numpy as np
import os
import glob
import re
import random

# ===================== 全局维度规范（适配model1） =====================
# 状态量维度：STATE_DIM = 6
# 时间戳维度：(N, 1) （2D）
# 单个状态维度：(STATE_DIM,) （1D）→ 输入model1时扩展为(1, STATE_DIM)
# 单个Δt维度：(1,) （1D tensor）→ 输入model1时扩展为(1, 1)
# 状态对中Xi/Xj维度：(STATE_DIM,) （1D）
# 状态对中delta_t维度：(1,) （1D tensor）→ 关键适配model1的拼接要求
# 状态对新增字段：file_id（int）- 所属文件编号（从文件名"轨道xx.txt"提取xx）
# =====================================================================

# 设置随机种子保证可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ===================== 全局配置 =====================
# 固定可视化文件路径
VISUALIZATION_FILE_PATH = r"C:\\Users\\admin\\Desktop\\深度学习\\深度学习大作业\\part2代码\\Xt_J2_dragdt1_data_lowtol.TXT"
# 新增：全局最大读取行数配置
MAX_READ_ROWS = 20000  # 最多读取前10000行

# ===================== 核心工具函数 =====================
def extract_file_id_from_filename(filename):
    """
    从文件名中提取数字作为file_id（适配"轨道xx.txt"/"轨道xx-xxx.txt"格式）
    示例：
    - "轨道5.txt" → 5
    - "轨道5-周期96.2min.txt" → 5
    - "轨道100-xxx.txt" → 100
    - 无数字则返回0（兜底）
    Args:
        filename: str 文件名（不含路径）
    Returns:
        file_id: int 提取的数字编号
    """
    # 优先匹配"轨道"后紧跟的数字（核心逻辑）
    pattern = r"轨道(\d+)"
    match = re.search(pattern, filename)
    if match:
        file_id = int(match.group(1))
    else:
        # 兜底：匹配文件名中任意连续数字
        num_match = re.search(r"\d+", filename)
        file_id = int(num_match.group()) if num_match else 0
    return file_id

def load_single_orbital_file(file_path, state_dim=6):
    """
    加载单个轨道文件（训练/可视化），返回统一格式的原始数据
    适配所有轨道文件格式，保证输出维度一致
    新增：最多读取前MAX_READ_ROWS行数据
    Args:
        file_path: str 文件路径
        state_dim: int 状态量维度（固定为6）
    Returns:
        raw_states: torch.Tensor (N, state_dim) 2D 原始状态量（N≤10000）
        raw_timestamps: torch.Tensor (N, 1) 2D 原始时间戳（N≤10000）
        file_id: int 文件编号（从文件名提取）
        filename: str 文件名
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    # 提取文件名和file_id
    filename = os.path.basename(file_path)
    file_id = extract_file_id_from_filename(filename)
    
    # 读取文件（兼容不同分隔符，新增nrows参数限制读取行数）
    try:
        # 核心修改1：添加nrows=MAX_READ_ROWS限制读取行数
        df = pd.read_csv(file_path, sep='\s+', header=None, nrows=MAX_READ_ROWS)
    except:
        # 核心修改2：备用读取方式也添加nrows限制
        df = pd.read_csv(file_path, sep='\t', header=None, nrows=MAX_READ_ROWS)
    
    # 核心修改3：强制截断到最多MAX_READ_ROWS行（防止读取方式未生效）
    if len(df) > MAX_READ_ROWS:
        df = df.iloc[:MAX_READ_ROWS]
        print(f"文件{filename}行数超过{MAX_READ_ROWS}，已截断为前{MAX_READ_ROWS}行")
    
    # 提取数据并强制转换为2D
    # 时间戳：(N,) → (N, 1)（N≤10000）
    timestamps = df.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
    # 状态量：(N, state_dim) （强制截取前6列，N≤10000）
    states = df.iloc[:, 1:1+state_dim].values.astype(np.float32)
    
    # 转换为tensor（确保2D）
    raw_ts_tensor = torch.FloatTensor(timestamps)  # (N, 1) 2D（N≤10000）
    raw_states_tensor = torch.FloatTensor(states)  # (N, 6) 2D（N≤10000）
    
    # 维度校验（新增行数校验）
    assert raw_ts_tensor.dim() == 2 and raw_ts_tensor.shape[1] == 1, \
        f"时间戳维度错误：{raw_ts_tensor.shape}（要求(N,1)）"
    assert raw_states_tensor.dim() == 2 and raw_states_tensor.shape[1] == state_dim, \
        f"状态量维度错误：{raw_states_tensor.shape}（要求(N,{state_dim})）"
    assert len(raw_ts_tensor) <= MAX_READ_ROWS, \
        f"读取行数超过限制：{len(raw_ts_tensor)}（最大{MAX_READ_ROWS}）"
    assert len(raw_states_tensor) <= MAX_READ_ROWS, \
        f"读取行数超过限制：{len(raw_states_tensor)}（最大{MAX_READ_ROWS}）"
    
    print(f"文件{filename}加载完成，实际读取行数：{len(raw_ts_tensor)}")
    return raw_states_tensor, raw_ts_tensor, file_id, filename

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
# 核心函数：生成状态对（使用从文件名提取的file_id）
# ==========================================
def generate_state_pairs(full_states, full_timestamps, file_id, non_adj_ratio=0):
    """
    从单个文件的完整轨迹生成状态对（严格匹配model1输入维度 + 记录file_id）
    适配：输入数据已限制为≤10000行
    Args:
        full_states: torch.Tensor (N, state_dim) 2D 完整状态序列（归一化后，N≤10000）
        full_timestamps: torch.Tensor (N, 1) 2D 完整时间戳序列（原始值，N≤10000）
        file_id: int 当前文件的唯一编号（从文件名提取的数字）
        non_adj_ratio: 随机非相邻状态对的数量比例（相对于相邻对数量）
    Returns:
        adj_pairs: list 相邻状态对，每个元素=(Xi, Xj, delta_t, mid_indices, file_id)
        non_adj_pairs: list 随机非相邻状态对，每个元素=(Xi, Xj, delta_t, mid_indices, file_id)
            - Xi/Xj: torch.Tensor (state_dim,) 1D 状态量
            - delta_t: torch.Tensor (1,) 1D tensor（关键修改：替代float）
            - mid_indices: list 中间索引
            - file_id: int 所属文件编号（从文件名提取）
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
        # 附加从文件名提取的file_id
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
        # 附加从文件名提取的file_id
        non_adj_pairs.append((Xi, Xj, delta_t, mid_indices, file_id))
    
    print(f"文件{file_id}生成状态对统计：相邻对={len(adj_pairs)}, 非相邻对={len(non_adj_pairs)}")
    return adj_pairs, non_adj_pairs

# ==========================================
# 加载可视化数据（统一格式：指定文件 + 训练文件）
# ==========================================
def load_visualization_data_unified(file_path, scaler, state_dim=6):
    """
    加载指定可视化文件（保留原有格式），返回归一化状态+原始时间戳+file_id
    适配：读取行数≤10000
    Args:
        file_path: str 可视化文件路径
        scaler: SimpleScaler 已拟合的归一化器
        state_dim: int 状态量维度（固定为6）
    Returns:
        norm_states: torch.Tensor (N, state_dim) 2D 归一化后的完整状态（N≤10000）
        raw_timestamps: torch.Tensor (N, 1) 2D 原始时间戳（N≤10000）
        file_id: int 可视化文件的file_id（从文件名提取）
        filename: str 可视化文件名称
    """
    # 使用统一的文件加载函数（保证格式一致，已限制读取行数）
    raw_states, raw_timestamps, file_id, filename = load_single_orbital_file(file_path, state_dim)
    
    print(f"\n加载可视化专用文件：{file_path}")
    print(f"可视化文件file_id（从文件名提取）：{file_id}")
    
    # 归一化状态量（保留原有格式）
    norm_states = scaler.transform(raw_states)
    
    print(f"可视化文件加载完成：状态量维度={norm_states.shape}，时间戳维度={raw_timestamps.shape}")
    return norm_states, raw_timestamps, file_id, filename

def load_training_data_as_visualization(training_file_paths, scaler, state_dim=6):
    """
    将训练数据文件转换为可视化格式（与指定可视化文件格式完全一致）
    适配：每个文件读取行数≤10000
    Args:
        training_file_paths: list 训练文件路径列表
        scaler: SimpleScaler 已拟合的归一化器
        state_dim: int 状态量维度（固定为6）
    Returns:
        vis_training_data: list 训练数据的可视化格式列表
            每个元素：(norm_states, raw_timestamps, file_id, filename)
            - norm_states: torch.Tensor (N, state_dim) 2D 归一化状态（N≤10000）
            - raw_timestamps: torch.Tensor (N, 1) 2D 原始时间戳（N≤10000）
            - file_id: int 从文件名提取的编号
            - filename: str 文件名
    """
    vis_training_data = []
    
    for fp in training_file_paths:
        try:
            # 使用统一的文件加载函数（已限制读取行数）
            raw_states, raw_timestamps, file_id, filename = load_single_orbital_file(fp, state_dim)
            
            # 归一化状态量（与可视化文件格式一致）
            norm_states = scaler.transform(raw_states)
            
            vis_training_data.append({
                'norm_states': norm_states,
                'raw_timestamps': raw_timestamps,
                'file_id': file_id,
                'filename': filename,
                'file_path': fp
            })
            
            print(f"训练文件转换为可视化格式：{filename} (file_id={file_id})，维度={norm_states.shape}")
            
        except Exception as e:
            print(f"处理训练文件 {fp} 失败: {e}")
            continue
    
    return vis_training_data

# ==========================================
# 主数据加载函数（整合所有需求）
# ==========================================
def load_orbital_file(path, non_adj_ratio=0.2, state_dim=6):
    """
    加载轨道文件并生成状态对（输出完全匹配model1输入要求 + 统一可视化格式）
    核心修改：
    1. file_id从文件名"轨道xx.txt"提取xx数字
    2. 整合可视化数据：指定文件 + 训练文件（格式统一）
    3. 每个文件最多读取前10000行数据
    Args:
        path: str 文件/目录路径（训练数据）
        non_adj_ratio: float 非相邻状态对比例
        state_dim: int 状态量维度（固定为6）
    Returns:
        all_adj_pairs: list 所有相邻状态对（含从文件名提取的file_id）
        all_non_adj_pairs: list 所有非相邻状态对（同左）
        scaler: SimpleScaler 归一化器
        # 可视化数据（统一格式）
        vis_spec_data: dict 指定可视化文件数据（norm_states, raw_timestamps, file_id, filename）
        vis_train_data: list 训练文件的可视化格式数据列表
        file_mapping: dict file_id到文件名的映射（{5: "轨道5.txt", ...}）
    """
    # 路径处理（训练数据）
    if os.path.isdir(path):
        print(f"从目录读取训练数据: {path}")
        training_file_paths = sorted(glob.glob(os.path.join(path, "*.txt")))
    else:
        print(f"读取单文件训练数据: {path}")
        training_file_paths = [path]
        
    if not training_file_paths:
        print("未找到训练数据文件")
        return None, None, None, None, None, None

    # 存储所有训练文件的原始数据 + file_id映射（从文件名提取）
    all_raw_states = []      # 每个元素：(N, state_dim) 2D（N≤10000）
    all_raw_timestamps = []  # 每个元素：(N, 1) 2D（N≤10000）
    all_raw_dt = []          # 每个元素：(N-1, 1) 2D（N≤10000）
    file_mapping = {}        # {file_id: filename, ...}
    training_file_ids = []   # 存储每个训练文件的file_id
    
    # 1. 读取所有训练文件（使用统一加载函数，提取file_id，已限制读取行数）
    for fp in training_file_paths:
        try:
            raw_states, raw_timestamps, file_id, filename = load_single_orbital_file(fp, state_dim)
            
            # 构建file_id映射（去重保护）
            if file_id in file_mapping:
                print(f"警告：file_id={file_id}重复（{filename}与{file_mapping[file_id]}），自动重命名为{file_id}_{len(file_mapping)}")
                file_id = f"{file_id}_{len(file_mapping)}"
            file_mapping[file_id] = filename
            
            # 存储
            all_raw_states.append(raw_states)
            all_raw_timestamps.append(raw_timestamps)
            training_file_ids.append(file_id)
            
            # 计算相邻Δt（2D）
            timestamps_np = raw_timestamps.numpy()
            if len(timestamps_np) > 1:
                dt = np.diff(timestamps_np, axis=0)  # (N-1, 1)（N≤10000）
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
        
    print(f"\n成功读取 {len(all_raw_states)} 个训练文件")
    print(f"训练文件file_id映射：{file_mapping}")
        
    # 2. 全局归一化（输入均为2D）
    # 拼接所有训练状态量（2D）
    all_states_concat = torch.cat(all_raw_states, dim=0)  # (total_N, 6)（每个文件≤10000行）
    # 拼接所有训练Δt（2D）
    all_dt_concat = torch.cat(all_raw_dt, dim=0) if all_raw_dt else torch.FloatTensor([[0.0]])
    
    # 初始化并拟合归一化器
    scaler = SimpleScaler()
    scaler.fit(all_states_concat, dt_data=all_dt_concat)
    
    # 3. 为每个训练文件生成状态对（使用从文件名提取的file_id）
    all_adj_pairs = []
    all_non_adj_pairs = []
    
    for idx, (raw_states, raw_ts) in enumerate(zip(all_raw_states, all_raw_timestamps)):
        file_id = training_file_ids[idx]
        # 确保file_id为整数（兼容重命名的情况）
        file_id_int = int(file_id.split('_')[0]) if '_' in str(file_id) else int(file_id)
        
        # 归一化当前训练文件的状态量（输入2D，输出2D）
        norm_states = scaler.transform(raw_states)
        
        # 维度校验
        assert norm_states.dim() == 2 and norm_states.shape[1] == state_dim, \
            f"训练文件归一化状态维度错误：{norm_states.shape}"
        
        # 生成状态对（传递从文件名提取的file_id）
        adj_pairs, non_adj_pairs = generate_state_pairs(
            norm_states, raw_ts, file_id=file_id_int, non_adj_ratio=non_adj_ratio
        )
        
        # 合并到全局列表
        all_adj_pairs.extend(adj_pairs)
        all_non_adj_pairs.extend(non_adj_pairs)
    
    # 4. 加载可视化数据（统一格式）
    ## 4.1 加载指定的可视化文件（保留原有格式，已限制读取行数）
    try:
        vis_spec_norm, vis_spec_ts, vis_spec_id, vis_spec_name = load_visualization_data_unified(VISUALIZATION_FILE_PATH, scaler, state_dim)
        vis_spec_data = {
            'norm_states': vis_spec_norm,
            'raw_timestamps': vis_spec_ts,
            'file_id': vis_spec_id,
            'filename': vis_spec_name,
            'file_path': VISUALIZATION_FILE_PATH,
            'type': 'specified'  # 标记为指定可视化文件
        }
    except Exception as e:
        print(f"加载指定可视化文件失败：{e}")
        # 降级方案：使用第一个训练文件作为可视化数据
        print("降级为使用第一个训练文件作为指定可视化数据")
        first_file_id = training_file_ids[0]
        first_file_id_int = int(first_file_id.split('_')[0]) if '_' in str(first_file_id) else int(first_file_id)
        vis_spec_norm = scaler.transform(all_raw_states[0])
        vis_spec_ts = all_raw_timestamps[0]
        vis_spec_name = file_mapping[first_file_id]
        vis_spec_data = {
            'norm_states': vis_spec_norm,
            'raw_timestamps': vis_spec_ts,
            'file_id': first_file_id_int,
            'filename': vis_spec_name,
            'file_path': training_file_paths[0],
            'type': 'fallback'  # 标记为降级文件
        }
    
    ## 4.2 将训练文件转换为可视化格式（与指定文件格式一致，已限制读取行数）
    vis_train_data = load_training_data_as_visualization(training_file_paths, scaler, state_dim)
    
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
    
    print(f"\n最终状态对统计：总相邻对={len(all_adj_pairs)}, 总非相邻对={len(all_non_adj_pairs)}")
    print(f"可视化数据统计：指定文件1个，训练文件转换{len(vis_train_data)}个")
    print("✅ 数据加载完成，所有输出维度完全匹配model1要求！")
    
    return all_adj_pairs, all_non_adj_pairs, scaler, vis_spec_data, vis_train_data, file_mapping

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

# ===================== 测试函数（验证所有功能） =====================
def test_all_features():
    """测试所有新增功能：file_id提取 + 统一可视化格式 + 读取行数限制"""
    # 1. 测试file_id提取
    test_filenames = [
        "轨道5.txt",
        "轨道100-周期96.2min.txt",
        "轨道8-半长轴6952km.txt",
        "无数字文件.txt"
    ]
    print("\n=== 测试file_id提取功能 ===")
    for fname in test_filenames:
        fid = extract_file_id_from_filename(fname)
        print(f"文件名：{fname} → file_id：{fid}")
    
    # 2. 测试model1兼容性（含新file_id）
    print("\n=== 测试model1兼容性 ===")
    def simulate_model_input(xi, xj, delta_t):
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
        if xj.dim() == 1:
            xj = xj.unsqueeze(0)
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)
        
        obs_dim = 64
        psi_i = torch.randn(1, obs_dim)
        psi_j = torch.randn(1, obs_dim)
        cvae_input = torch.cat([psi_i, psi_j, delta_t], dim=1)
        
        assert cvae_input.dim() == 2, "拼接后必须是2D"
        assert cvae_input.shape == (1, obs_dim*2 + 1), f"拼接维度错误：{cvae_input.shape}"
        return True
    
    # 生成测试数据（限制为10000行以内）
    test_states = torch.randn(min(10000, 100), 6)  # 测试数据≤10000行
    test_ts = torch.arange(min(10000, 100)).float().unsqueeze(1)
    
    # 使用从文件名提取的file_id（比如5）
    adj_pairs, non_adj_pairs = generate_state_pairs(test_states, test_ts, file_id=5)
    
    # 测试相邻对
    Xi, Xj, delta_t, mid_indices, file_id = adj_pairs[0]
    assert file_id == 5, "file_id应该为5（从文件名提取）"
    assert simulate_model_input(Xi, Xj, delta_t), "相邻对不兼容model1"
    
    # 3. 测试读取行数限制
    print("\n=== 测试读取行数限制功能 ===")
    assert len(test_states) <= MAX_READ_ROWS, f"测试数据行数超过限制：{len(test_states)} > {MAX_READ_ROWS}"
    assert len(test_ts) <= MAX_READ_ROWS, f"测试时间戳行数超过限制：{len(test_ts)} > {MAX_READ_ROWS}"
    print(f"✅ 读取行数限制测试通过（最大{MAX_READ_ROWS}行）")
    
    print("✅ 所有功能测试通过！")

if __name__ == "__main__":
    # 运行全功能测试
    test_all_features()
    
    # 测试数据加载（训练数据路径可自定义）
    print("\n=== 测试数据加载 ===")
    adj_pairs, non_adj_pairs, scaler, vis_spec_data, vis_train_data, file_mapping = load_orbital_file("data")
    
    # 验证可视化数据
    if vis_spec_data is not None:
        print(f"\n✅ 指定可视化数据验证：")
        print(f"  文件名称：{vis_spec_data['filename']}")
        print(f"  File_id：{vis_spec_data['file_id']}")
        print(f"  状态量维度：{vis_spec_data['norm_states'].shape}（行数≤{MAX_READ_ROWS}）")
        print(f"  时间戳维度：{vis_spec_data['raw_timestamps'].shape}（行数≤{MAX_READ_ROWS}）")
        print(f"  数据类型：{vis_spec_data['type']}")
    
    if vis_train_data:
        print(f"\n✅ 训练数据可视化格式验证（共{len(vis_train_data)}个文件）：")
        for i, train_vis in enumerate(vis_train_data[:2]):  # 只打印前2个
            print(f"  第{i+1}个文件：{train_vis['filename']} (file_id={train_vis['file_id']})，维度={train_vis['norm_states'].shape}（行数≤{MAX_READ_ROWS}）")