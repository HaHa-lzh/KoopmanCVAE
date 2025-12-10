import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 核心模型结构（CVAE+Koopman微分算子+标签化监督）
# 核心修改：CVAE标签从(Xi, Δt)改为(file_id, Δt)
# ==========================================
class KoopmanCVAE(nn.Module):  # 重命名为KoopmanCVAE，体现CVAE核心
    def __init__(self, state_dim, obs_dim, k_latent_dim, hidden_dim=128, max_file_id=100):
        """
        重构后核心逻辑：
        1. 编码器：X→Psi（无修改）
        2. 算子推理（CVAE编码器）：(psi_i, psi_{i+1}, Δt_i) → 微分算子分布(mu, log_var)
        3. CVAE标签：(file_id, Δt_i) 作为生成网络的条件标签（核心修改）
        4. 算子生成（CVAE解码器）：(z, 标签) → Koopman算子矩阵K
        5. 重构约束：K * psi_i ≈ psi_{i+1}
        
        Args:
            state_dim (int): 原始状态X维度（轨道6维）
            obs_dim (int): Koopman观测空间Psi维度
            k_latent_dim (int): Koopman微分算子潜在维度
            hidden_dim (int): 隐藏层神经元数量
            max_file_id (int): 文件ID的最大取值（用于Embedding层）
        """
        super(KoopmanCVAE, self).__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.k_latent_dim = k_latent_dim
        self.max_file_id = max_file_id
        
        # ========== 核心新增：File ID Embedding层 ==========
        # 将离散的file_id转换为连续向量（适配CVAE解码器输入）
        self.file_id_embedding = nn.Embedding(
            num_embeddings=max_file_id + 1,  # ID范围：0 ~ max_file_id
            embedding_dim=32  # 嵌入维度（可根据需求调整）
        )
        self.embedding_dim = 32  # 记录嵌入维度
        
        # 1. 编码器 F: X → Psi（完全保留原有结构）
        self.encoder_f = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # 2. CVAE算子推理网络（编码器）：(psi_i, psi_{i+1}, Δt) → (mu, log_var)
        # 输入：psi_i + psi_{i+1} + Δt（转移对三要素）
        self.cvae_encoder = nn.Sequential(
            nn.Linear(obs_dim * 2 + 1, hidden_dim * 2),  # obs_dim*2=psi_i+psi_{i+1}, +1=Δt
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, k_latent_dim * 2)  # 输出微分算子分布参数
        )
        
        # 3. CVAE算子生成网络（解码器）：z + 标签(file_id_emb, Δt) → Koopman矩阵
        # 标签维度：embedding_dim(file_id) + 1(Δt) （核心修改）
        self.cvae_decoder = nn.Sequential(
            nn.Linear(k_latent_dim + self.embedding_dim + 1, hidden_dim * 2),  # z + 标签
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obs_dim * obs_dim)  # 输出Koopman矩阵展平
        )
        
        # 4. 解码器 F': Psi → X（完全保留原有结构）
        self.decoder_f_prime = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def reparameterize(self, mu, log_var):
        """重参数化技巧（无修改）"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_koopman_matrix(self, z, file_id, delta_t):
        """
        CVAE生成Koopman算子矩阵（核心修改：标签改为file_id + Δt）
        Args:
            z: (batch_size, k_latent_dim) 重采样的潜在特征
            file_id: (batch_size,) LongTensor - 文件ID（离散整数）
            delta_t: (batch_size, 1) FloatTensor - 时间间隔
        Returns:
            K_matrix: (batch_size, obs_dim, obs_dim) Koopman算子矩阵
        """
        batch_size = z.size(0)
        
        # ========== 核心修改：处理file_id并构建新标签 ==========
        # 1. File ID嵌入（离散→连续）
        file_id_emb = self.file_id_embedding(file_id)  # (batch_size, embedding_dim)
        
        # 2. 构建CVAE标签：file_id_emb + Δt
        label = torch.cat([file_id_emb, delta_t], dim=1)  # (batch_size, embedding_dim + 1)
        
        # 3. 拼接潜在特征和标签
        z_with_label = torch.cat([z, label], dim=1)  # (batch_size, k_latent_dim + embedding_dim + 1)
        
        # 4. 生成Koopman矩阵
        k_flat = self.cvae_decoder(z_with_label)
        K_matrix = k_flat.view(batch_size, self.obs_dim, self.obs_dim)
        
        return K_matrix
    
    # ===================== 新增方法：计算K矩阵逐次乘积（适配非相邻状态对） =====================
    def compute_k_product(self, k_matrices_list):
        """
        批量计算多步K矩阵的逐次乘积（用于非相邻状态对的多步误差计算）
        Args:
            k_matrices_list: list of tensor，每个tensor形状为(batch_size, obs_dim, obs_dim)
                            对应(Xi,Xi+1),(Xi+1,Xi+2)...(Xj-1,Xj)的K矩阵
        Returns:
            K_product: (batch_size, obs_dim, obs_dim) 逐次乘积结果 K_i * K_{i+1} * ... * K_{j-1}
        """
        if not k_matrices_list:  # 空列表兜底
            return torch.eye(self.obs_dim).unsqueeze(0).repeat(
                1, 1, 1).to(next(self.parameters()).device)
        
        # 初始化乘积为单位矩阵（匹配batch_size）
        batch_size = k_matrices_list[0].shape[0]
        K_product = torch.eye(self.obs_dim).unsqueeze(0).repeat(
            batch_size, 1, 1).to(k_matrices_list[0].device)
        
        # 逐次矩阵乘法（按顺序相乘）
        for K in k_matrices_list:
            K_product = torch.bmm(K_product, K)
        
        return K_product

    def forward_single_pair(self, x_i, x_i_plus_1, delta_t, file_id):
        """
        单转移对前向（核心修改：新增file_id输入）
        Args:
            x_i: (batch_size, state_dim) 前一时刻状态
            x_i_plus_1: (batch_size, state_dim) 后一时刻状态
            delta_t: (batch_size, 1) 时间间隔
            file_id: (batch_size,) LongTensor - 每个样本对应的文件ID
        Returns:
            psi_i: X_i的编码
            psi_i_plus_1: X_{i+1}的编码
            K_matrix: 对应标签的Koopman算子
            psi_recon: K*psi_i（重构后psi_{i+1}）
            x_i_recon: psi_i解码回X_i
            x_i_plus_1_recon: psi_recon解码回X_{i+1}
            mu/log_var: 微分算子分布参数
        """
        # ========== 强制统一所有输入维度 ==========
        # 确保状态量是2D (B, 6)
        if x_i.dim() == 1:
            x_i = x_i.unsqueeze(0)  # 单样本：(6,) → (1,6)
        if x_i_plus_1.dim() == 1:
            x_i_plus_1 = x_i_plus_1.unsqueeze(0)  # (6,) → (1,6)
        
        # 确保delta_t是2D (B, 1)
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(1)  # (B,) → (B,1) 或 (1,) → (1,1)
        
        # 确保file_id是1D LongTensor (B,)
        if file_id.dim() > 1:
            file_id = file_id.squeeze(-1)  # 去除多余维度
        if not file_id.dtype == torch.long:
            file_id = file_id.long()
        
        # ========== 原有逻辑：编码状态量 ==========
        psi_i = self.encoder_f(x_i)          # 2D (B, obs_dim)
        psi_i_plus_1 = self.encoder_f(x_i_plus_1)  # 2D (B, obs_dim)
        
        # ========== CVAE编码器：推理微分算子分布 ==========
        cvae_input = torch.cat([psi_i, psi_i_plus_1, delta_t], dim=1)
        k_params = self.cvae_encoder(cvae_input)
        mu, log_var = torch.chunk(k_params, 2, dim=1)
        
        # ========== 重采样潜在特征z ==========
        z = self.reparameterize(mu, log_var)
        
        # ========== 核心修改：生成Koopman算子（使用file_id + Δt作为标签） ==========
        K_matrix = self.get_koopman_matrix(z, file_id, delta_t)
        
        # ========== 单步重构与解码 ==========
        # 1. 单步重构：K*psi_i ≈ psi_{i+1}
        psi_recon = torch.bmm(K_matrix, psi_i.unsqueeze(2)).squeeze(2)
        
        # 2. 解码回X空间
        x_i_recon = self.decoder_f_prime(psi_i)
        x_i_plus_1_recon = self.decoder_f_prime(psi_recon)
        
        return psi_i, psi_i_plus_1, K_matrix, psi_recon, x_i_recon, x_i_plus_1_recon, mu, log_var

    def predict_long_term(self, x0, delta_t_seq, file_id, steps, use_prior=True, z_mean=None):
        """
        修正后长时预测（核心修改：新增file_id输入）
        Args:
            x0: (1, state_dim) 初始状态（batch_size=1）
            delta_t_seq: (N, 1) 时间间隔序列（numpy/tensor）
            file_id: int/LongTensor - 预测使用的文件ID（单个值）
            steps: int 预测步数
            use_prior: bool 是否使用先验分布（N(0,1)）采样z，False则用输入的z_mean
            z_mean: (1, k_latent_dim) 自定义潜在特征（如训练时的均值）
        Returns:
            trajectory: (steps+1, state_dim) 预测轨迹
            K_matrices: list 每步的Koopman算子
        """
        self.eval()
        # 核心：提前获取基准设备（从x0提取，确保所有张量统一）
        device = x0.device
        trajectory = [x0[0].cpu().numpy()]  # 初始状态
        K_matrices = []

        print("\n" + "="*50)
        print("🔍 delta_t_seq 序列详细信息")
        print("="*50)
        
        # 1. 处理Δt序列
        if isinstance(delta_t_seq, np.ndarray):
            print(f"原始类型：numpy.ndarray | 原始形状：{delta_t_seq.shape}")
            delta_t_seq_tensor = torch.FloatTensor(delta_t_seq).to(device)
        elif isinstance(delta_t_seq, torch.Tensor):
            print(f"原始类型：torch.Tensor | 原始设备：{delta_t_seq.device} | 原始形状：{delta_t_seq.shape}")
            delta_t_seq_tensor = delta_t_seq.to(device)
        else:
            print(f"原始类型：{type(delta_t_seq)}（不支持）")
            delta_t_seq_tensor = torch.tensor([], device=device)
        
        # 2. 处理file_id（确保是1D LongTensor）
        if isinstance(file_id, int):
            file_id_tensor = torch.tensor([file_id], dtype=torch.long, device=device)
        else:
            file_id_tensor = file_id.to(device).long()
            if file_id_tensor.dim() == 0:
                file_id_tensor = file_id_tensor.unsqueeze(0)
        
        # 3. 统一格式后的信息
        if len(delta_t_seq_tensor) > 0:
            delta_t_seq_flat = delta_t_seq_tensor.cpu().numpy().flatten()
            print(f"统一后形状：{delta_t_seq_tensor.shape} | 展平后长度：{len(delta_t_seq_flat)}")
            print(f"预测步数：{steps} | Δt序列长度是否匹配：{'✅' if len(delta_t_seq_flat) >= steps else '❌'}")
            
            # 打印前100个值
            print("\n📊 前100个Δt值（每行10个）：")
            print_limit = min(100, len(delta_t_seq_flat))
            delta_t_print = delta_t_seq_flat[:print_limit]
            for i in range(0, print_limit, 10):
                end_idx = min(i+10, print_limit)
                dt_line = [f"{dt:.6f}" for dt in delta_t_print[i:end_idx]]
                print(f"  第{i+1}-{end_idx}个：{', '.join(dt_line)}")
        else:
            print("❌ Δt序列为空！")
        print("="*50 + "\n")

        x_curr = x0.to(device)  # 初始状态
        
        with torch.no_grad():
            for i in range(steps):
                # 1. 获取当前步的Δt
                take_len = min(i, len(delta_t_seq))
                delta_t_prev = delta_t_seq_tensor[:take_len] if take_len > 0 else torch.tensor([[0.0]], device=device)
                delta_t_sum = delta_t_prev.sum().item()
                delta_t = torch.tensor([[delta_t_sum]], dtype=torch.float32, device=device)
                
                # 2. 潜在空间采样
                if use_prior:
                    z = torch.randn(1, self.k_latent_dim, dtype=torch.float32, device=device)
                else:
                    if z_mean is None:
                        z = torch.zeros(1, self.k_latent_dim, dtype=torch.float32, device=device)
                    else:
                        z = z_mean.to(device)
                
                # 3. 核心修改：使用file_id + Δt生成Koopman算子
                K_matrix = self.get_koopman_matrix(z, file_id_tensor, delta_t)  # (1, obs_dim, obs_dim)
                K_matrices.append(K_matrix.cpu().numpy())
                
                # 4. Koopman算子映射：psi_curr → psi_next
                psi_curr = self.encoder_f(x_curr)  # (1, obs_dim)
                psi_next = torch.bmm(K_matrix, psi_curr.unsqueeze(2)).squeeze(2)  # (1, obs_dim)
                
                # 5. 解码回X空间
                x_next = self.decoder_f_prime(psi_next)  # (1, state_dim)
                
                # 6. 更新轨迹和当前状态
                trajectory.append(x_next[0].cpu().numpy())
                #x_curr = x_next
        
        return np.array(trajectory), K_matrices
    

#  ==========================================
# 损失函数（适配标签修改，保留按文件分组的约束）
# ==========================================
def loss_cvae_koopman(psi_recon, psi_true, 
                      x_i_recon, x_i_true,
                      x_i_plus_1_recon, x_i_plus_1_true,
                      mu, log_var, K_matrix,
                      file_ids,  # 每个样本对应的文件编号 (batch_size,)
                      K_product=None, lambda_k_product=10.0,
                      alpha=1.0, beta=0.1, gamma=1, theta=0.1, delta=1.0):
    """
    损失函数（适配标签修改，保留原有分组约束逻辑）
    1. psi_recon_loss: K*psi_i ≈ psi_{i+1}（核心单步重构）
    2. x_recon_loss: X重构损失
    3. kl_loss: CVAE潜在分布KL散度
    4. k_product_loss: 多步K乘积误差（仅K_product不为None时生效）
    5. latent_consistency_loss: 按文件分组约束mu/log_var（同文件内样本接近）
    
    Args:
        file_ids: (batch_size,) tensor - 每个样本对应的文件编号（如0,0,1,1,2...）
        K_product: (batch_size, obs_dim, obs_dim) 多步K矩阵乘积
        lambda_k_product: 多步K乘积误差的权重
        delta: 按文件分组的一致性损失权重
    """
    # 1. 核心重构损失：K*psi_i ≈ psi_{i+1}
    psi_recon_loss = F.mse_loss(psi_recon, psi_true)
    
    # 2. X空间重构损失
    x_i_recon_loss = F.mse_loss(x_i_recon, x_i_true)
    x_i_plus_1_recon_loss = F.mse_loss(x_i_plus_1_recon, x_i_plus_1_true)
    x_recon_loss = (x_i_recon_loss + x_i_plus_1_recon_loss) / 2
    
    # 3. CVAE KL散度损失
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / psi_recon.size(0)
    
    # 4. 多步K乘积误差（原有逻辑）
    k_product_loss = torch.tensor(0.0).to(psi_recon.device)
    if K_product is not None:
        k_product_loss = F.mse_loss(K_matrix, K_product) * lambda_k_product
    
    # 5. 按文件分组的mu/log_var一致性损失（原有逻辑，保留）
    latent_consistency_loss = torch.tensor(0.0).to(psi_recon.device)
    unique_file_ids = torch.unique(file_ids)
    
    for fid in unique_file_ids:
        mask = (file_ids == fid)
        if not torch.any(mask):
            continue
        
        mu_group = mu[mask]
        log_var_group = log_var[mask]
        
        mu_group_mean = mu_group.mean(dim=0, keepdim=True)
        log_var_group_mean = log_var_group.mean(dim=0, keepdim=True)
        
        mu_consistency = F.mse_loss(mu_group, mu_group_mean.expand_as(mu_group))
        log_var_consistency = F.mse_loss(log_var_group, log_var_group_mean.expand_as(log_var_group))
        
        n_group = torch.sum(mask).float()
        latent_consistency_loss += (mu_consistency + log_var_consistency) * n_group
    
    latent_consistency_loss = (latent_consistency_loss / mu.size(0)) * delta

    # 总损失
    total_loss = (psi_recon_loss * alpha + 
                  x_recon_loss * beta + 
                  kl_loss * gamma +
                  k_product_loss +
                  latent_consistency_loss)
    
    # 返回所有损失项
    return total_loss, psi_recon_loss, x_recon_loss, kl_loss, k_product_loss, latent_consistency_loss

