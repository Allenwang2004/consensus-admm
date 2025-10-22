"""
Consensus ADMM 算法解決 Lasso 回歸問題
將數據分散到多個節點，每個節點運行局部 ADMM，並通過全域變量協調

Consensus ADMM 分解 (對於 K 個節點):
Node k: minimize (1/2)||A_k x_k - b_k||^2 + λ||z||_1
全域約束: x_1 = x_2 = ... = x_K = z

擴展拉格朗日函数:
L = Σ_k [(1/2)||A_k x_k - b_k||^2 + u_k^T(x_k - z) + (ρ/2)||x_k - z||^2] + λ||z||_1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from copy import deepcopy
import time
warnings.filterwarnings('ignore')

# 設置中文字體支持
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class ConsensusADMMNode:
    """
    Consensus ADMM 的單個節點
    """
    def __init__(self, node_id, A_local, b_local, lambda_reg=0.1, rho=1.0):
        """
        初始化 Consensus ADMM 節點
        
        參數:
        node_id: 節點 ID
        A_local: 局部數據矩陣
        b_local: 局部目標向量
        lambda_reg: L1 正則化參數
        rho: 增廣拉格朗日乘數
        """
        self.node_id = node_id
        self.A_local = A_local
        self.b_local = b_local
        self.lambda_reg = lambda_reg
        self.rho = rho
        
        # 獲取維度
        self.m_local, self.n = A_local.shape
        
        # 初始化變量
        self.x_local = np.zeros(self.n)  # 局部變量
        self.u_local = np.zeros(self.n)  # 局部對偶變量
        
        # 預計算 (A_k^T A_k + ρI)^{-1}
        AtA = self.A_local.T @ self.A_local
        self.AtA_rhoI_inv = np.linalg.inv(AtA + self.rho * np.eye(self.n))
        self.Atb = self.A_local.T @ self.b_local
        
    def update_local_x(self, z_global):
        """
        更新局部變量 x_k
        x_k^{k+1} = (A_k^T A_k + ρI)^{-1} [A_k^T b_k + ρ(z - u_k)]
        """
        self.x_local = self.AtA_rhoI_inv @ (self.Atb + self.rho * (z_global - self.u_local))
        
    def update_local_u(self, z_global):
        """
        更新局部對偶變量 u_k
        u_k^{k+1} = u_k + x_k^{k+1} - z^{k+1}
        """
        self.u_local = self.u_local + self.x_local - z_global
        
    def compute_local_objective(self, z_global):
        """
        計算節點的局部目標函數值
        """
        return 0.5 * np.linalg.norm(self.A_local @ self.x_local - self.b_local)**2

class ConsensusADMMLasso:
    """
    Consensus ADMM Lasso 求解器
    """
    def __init__(self, lambda_reg=0.1, rho=1.0, max_iter=1000, tol=1e-6):
        """
        Consensus ADMM 求解器
        
        參數:
        lambda_reg: L1 正則化參數
        rho: 增廣拉格朗日乘數
        max_iter: 最大迭代次數
        tol: 收斂容忍度
        """
        self.lambda_reg = lambda_reg
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.nodes = []
        self.objective_history = []
        self.residual_history = []
        self.communication_rounds = 0
        
    def add_node(self, A_local, b_local):
        """
        添加一個節點到 Consensus ADMM
        """
        node_id = len(self.nodes)
        node = ConsensusADMMNode(node_id, A_local, b_local, self.lambda_reg, self.rho)
        self.nodes.append(node)
        return node_id
        
    def soft_threshold(self, x, threshold):
        """
        軟閾值函數
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
    def update_global_z(self):
        """
        更新全域變量 z
        z^{k+1} = soft_threshold(x̄ + ū, λ/ρK)
        其中 x̄ = (1/K) Σ_k x_k, ū = (1/K) Σ_k u_k
        """
        K = len(self.nodes)
        
        # 計算平均局部變量和對偶變量
        x_avg = np.mean([node.x_local for node in self.nodes], axis=0)
        u_avg = np.mean([node.u_local for node in self.nodes], axis=0)
        
        # 軟閾值更新
        z_global = self.soft_threshold(x_avg + u_avg, self.lambda_reg / (self.rho * K))
        
        return z_global
        
    def compute_global_objective(self, z_global):
        """
        計算全域目標函數值
        """
        total_obj = 0.0
        for node in self.nodes:
            total_obj += node.compute_local_objective(z_global)
        total_obj += self.lambda_reg * np.linalg.norm(z_global, 1)
        return total_obj
        
    def compute_residuals(self, z_global, z_global_old):
        """
        計算原始和對偶殘差
        """
        K = len(self.nodes)
        
        # 原始殘差: ||x_k - z||
        primal_residuals = [np.linalg.norm(node.x_local - z_global) for node in self.nodes]
        primal_residual = np.sqrt(np.sum([r**2 for r in primal_residuals]))
        
        # 對偶殘差: ρ||z^{k+1} - z^k||
        dual_residual = self.rho * np.sqrt(K) * np.linalg.norm(z_global - z_global_old)
        
        return primal_residual, dual_residual
        
    def fit(self, A_full=None, b_full=None, x_true=None, verbose=False):
        """
        使用 Consensus ADMM 求解 Lasso 問題
        """
        if len(self.nodes) == 0:
            raise ValueError("沒有添加任何節點!")
            
        K = len(self.nodes)
        n = self.nodes[0].n
        
        # 初始化全域變量
        z_global = np.zeros(n)
        
        # 清空歷史記錄
        self.objective_history = []
        self.residual_history = []
        self.communication_rounds = 0
        
        if verbose:
            print(f"開始 Consensus ADMM 迭代 (K = {K} 個節點)...")
            print(f"λ = {self.lambda_reg}, ρ = {self.rho}")
            print("-" * 70)
            
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            z_global_old = z_global.copy()
            
            # 步驟 1: 並行更新所有節點的局部變量 x_k
            for node in self.nodes:
                node.update_local_x(z_global)
            
            # 步驟 2: 更新全域變量 z (需要通信)
            z_global = self.update_global_z()
            self.communication_rounds += 1
            
            # 步驟 3: 更新所有節點的對偶變量 u_k
            for node in self.nodes:
                node.update_local_u(z_global)
            
            # 計算目標函數值
            obj_val = self.compute_global_objective(z_global)
            self.objective_history.append(obj_val)
            
            # 計算殘差
            primal_residual, dual_residual = self.compute_residuals(z_global, z_global_old)
            self.residual_history.append((primal_residual, dual_residual))
            
            # 檢查收斂
            if primal_residual < self.tol and dual_residual < self.tol:
                if verbose:
                    print(f"迭代 {iteration+1}: 收斂!")
                break
                
            # 打印進度
            if verbose and (iteration+1) % 50 == 0:
                print(f"迭代 {iteration+1:4d}: 目標函數 = {obj_val:.6f}, "
                      f"原始殘差 = {primal_residual:.2e}, "
                      f"對偶殘差 = {dual_residual:.2e}")
                if x_true is not None:
                    mse = np.mean((z_global - x_true)**2)
                    print(f"           MSE (vs 真實) = {mse:.6f}")
        
        self.total_time = time.time() - start_time
        
        if verbose:
            print("-" * 70)
            print(f"Consensus ADMM 完成! 總迭代次數: {iteration+1}")
            print(f"通信輪數: {self.communication_rounds}")
            print(f"總運行時間: {self.total_time:.4f} 秒")
            print(f"最終目標函數值: {self.objective_history[-1]:.6f}")
        
        self.z_final = z_global
        return z_global

def split_data(A, b, num_nodes, method='row', random_state=42):
    """
    將數據分割給多個節點
    
    參數:
    A: 完整設計矩陣
    b: 完整目標向量
    num_nodes: 節點數量
    method: 分割方法 ('row' 或 'random')
    random_state: 隨機種子
    """
    m, n = A.shape
    np.random.seed(random_state)
    
    if method == 'row':
        # 按行順序分割
        samples_per_node = m // num_nodes
        A_splits = []
        b_splits = []
        
        for k in range(num_nodes):
            start_idx = k * samples_per_node
            if k == num_nodes - 1:  # 最後一個節點包含剩餘樣本
                end_idx = m
            else:
                end_idx = (k + 1) * samples_per_node
                
            A_splits.append(A[start_idx:end_idx, :])
            b_splits.append(b[start_idx:end_idx])
            
    elif method == 'random':
        # 隨機分割
        indices = np.random.permutation(m)
        samples_per_node = m // num_nodes
        A_splits = []
        b_splits = []
        
        for k in range(num_nodes):
            start_idx = k * samples_per_node
            if k == num_nodes - 1:
                end_idx = m
            else:
                end_idx = (k + 1) * samples_per_node
                
            node_indices = indices[start_idx:end_idx]
            A_splits.append(A[node_indices, :])
            b_splits.append(b[node_indices])
    
    return A_splits, b_splits

# 重用之前的 ADMMLasso 類別和其他函數
class ADMMLasso:
    def __init__(self, lambda_reg=0.1, rho=1.0, max_iter=1000, tol=1e-6):
        self.lambda_reg = lambda_reg
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.objective_history = []
        self.residual_history = []
        
    def soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def compute_objective(self, A, b, x, z):
        return 0.5 * np.linalg.norm(A @ x - b)**2 + self.lambda_reg * np.linalg.norm(z, 1)
    
    def fit(self, A, b, x_true=None, verbose=False):
        m, n = A.shape
        x = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)
        
        AtA = A.T @ A
        Atb = A.T @ b
        AtA_rhoI_inv = np.linalg.inv(AtA + self.rho * np.eye(n))
        
        self.objective_history = []
        self.residual_history = []
        
        start_time = time.time()
        
        for k in range(self.max_iter):
            x_old = x.copy()
            z_old = z.copy()
            
            x = AtA_rhoI_inv @ (Atb + self.rho * (z - u))
            z = self.soft_threshold(x + u, self.lambda_reg / self.rho)
            u = u + x - z
            
            obj_val = self.compute_objective(A, b, x, z)
            self.objective_history.append(obj_val)
            
            primal_residual = np.linalg.norm(x - z)
            dual_residual = self.rho * np.linalg.norm(z - z_old)
            self.residual_history.append((primal_residual, dual_residual))
            
            if primal_residual < self.tol and dual_residual < self.tol:
                break
        
        self.total_time = time.time() - start_time
        self.x_final = x
        self.z_final = z
        return x

def generate_sparse_data(m=200, n=50, sparsity=0.1, noise_std=0.1, random_state=42):
    """生成稀疏回歸數據"""
    np.random.seed(random_state)
    
    A = np.random.randn(m, n)
    A = A / np.sqrt(np.sum(A**2, axis=0))
    
    x_true = np.zeros(n)
    n_nonzero = int(sparsity * n)
    nonzero_indices = np.random.choice(n, n_nonzero, replace=False)
    x_true[nonzero_indices] = np.random.randn(n_nonzero) * 2
    
    b = A @ x_true + noise_std * np.random.randn(m)
    
    return A, b, x_true

def plot_consensus_comparison(single_admm, consensus_admm, x_true, x_single, x_consensus, A, b):
    """
    比較單一 ADMM 和 Consensus ADMM 的結果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 目標函數收斂比較
    axes[0, 0].semilogy(single_admm.objective_history, label='單一 ADMM', alpha=0.8)
    axes[0, 0].semilogy(consensus_admm.objective_history, label='Consensus ADMM', alpha=0.8)
    axes[0, 0].set_title('目標函數收斂比較')
    axes[0, 0].set_xlabel('迭代次數')
    axes[0, 0].set_ylabel('目標函數值 (log scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 權重估計比較
    indices = np.arange(len(x_true))
    width = 0.25
    axes[0, 1].bar(indices - width, x_true, width, label='真實權重', alpha=0.7)
    axes[0, 1].bar(indices, x_single, width, label='單一 ADMM', alpha=0.7)
    axes[0, 1].bar(indices + width, x_consensus, width, label='Consensus ADMM', alpha=0.7)
    axes[0, 1].set_title('權重估計比較')
    axes[0, 1].set_xlabel('特徵索引')
    axes[0, 1].set_ylabel('權重值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 單一 vs Consensus 權重散點圖
    axes[0, 2].scatter(x_single, x_consensus, alpha=0.7)
    min_val = min(np.min(x_single), np.min(x_consensus))
    max_val = max(np.max(x_single), np.max(x_consensus))
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 2].set_title('單一 ADMM vs Consensus ADMM')
    axes[0, 2].set_xlabel('單一 ADMM 權重')
    axes[0, 2].set_ylabel('Consensus ADMM 權重')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 殘差比較
    single_primal = [r[0] for r in single_admm.residual_history]
    single_dual = [r[1] for r in single_admm.residual_history]
    consensus_primal = [r[0] for r in consensus_admm.residual_history]
    consensus_dual = [r[1] for r in consensus_admm.residual_history]
    
    axes[1, 0].semilogy(single_primal, label='單一 ADMM 原始', alpha=0.8, linestyle='-')
    axes[1, 0].semilogy(single_dual, label='單一 ADMM 對偶', alpha=0.8, linestyle='--')
    axes[1, 0].semilogy(consensus_primal, label='Consensus 原始', alpha=0.8, linestyle='-')
    axes[1, 0].semilogy(consensus_dual, label='Consensus 對偶', alpha=0.8, linestyle='--')
    axes[1, 0].set_title('殘差收斂比較')
    axes[1, 0].set_xlabel('迭代次數')
    axes[1, 0].set_ylabel('殘差 (log scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 預測性能比較
    b_pred_single = A @ x_single
    b_pred_consensus = A @ x_consensus
    
    axes[1, 1].scatter(b, b_pred_single, alpha=0.6, label='單一 ADMM', s=30)
    axes[1, 1].scatter(b, b_pred_consensus, alpha=0.6, label='Consensus ADMM', s=30)
    min_b = min(np.min(b), np.min(b_pred_single), np.min(b_pred_consensus))
    max_b = max(np.max(b), np.max(b_pred_single), np.max(b_pred_consensus))
    axes[1, 1].plot([min_b, max_b], [min_b, max_b], 'k--', alpha=0.8)
    axes[1, 1].set_title('預測值 vs 真實值')
    axes[1, 1].set_xlabel('真實 b')
    axes[1, 1].set_ylabel('預測 Ax')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 權重差異分析
    weight_diff = np.abs(x_single - x_consensus)
    axes[1, 2].bar(indices, weight_diff, alpha=0.7)
    axes[1, 2].set_title('權重差異 |單一 - Consensus|')
    axes[1, 2].set_xlabel('特徵索引')
    axes[1, 2].set_ylabel('絕對差異')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函數: 比較單一 ADMM 和 Consensus ADMM
    """
    print("=" * 80)
    print("Consensus ADMM vs 單一 ADMM 比較演示")
    print("=" * 80)
    
    # 生成合成數據
    print("\n1. 生成合成稀疏數據...")
    m, n = 200, 50  # 200 樣本, 50 特徵
    sparsity = 0.2
    A, b, x_true = generate_sparse_data(m, n, sparsity=sparsity, noise_std=0.1)
    
    print(f"   數據維度: {A.shape}")
    print(f"   稀疏度: {sparsity}")
    print(f"   真實非零係數數量: {np.sum(x_true != 0)}")
    
    # 設置參數
    lambda_reg = 0.1
    rho = 1.0
    num_nodes = 4  # Consensus ADMM 節點數
    
    print(f"\n2. 算法參數設置:")
    print(f"   λ (正則化參數) = {lambda_reg}")
    print(f"   ρ (增廣參數) = {rho}")
    print(f"   Consensus ADMM 節點數 = {num_nodes}")
    
    # 運行單一 ADMM
    print(f"\n3. 運行單一 ADMM...")
    single_admm = ADMMLasso(lambda_reg=lambda_reg, rho=rho, max_iter=1000, tol=1e-6)
    x_single = single_admm.fit(A, b, x_true=x_true, verbose=True)
    
    # 準備 Consensus ADMM 數據
    print(f"\n4. 準備 Consensus ADMM 數據分割...")
    A_splits, b_splits = split_data(A, b, num_nodes, method='row')
    
    print("   數據分割結果:")
    for i, (A_i, b_i) in enumerate(zip(A_splits, b_splits)):
        print(f"     節點 {i}: {A_i.shape[0]} 樣本")
    
    # 運行 Consensus ADMM
    print(f"\n5. 運行 Consensus ADMM...")
    consensus_admm = ConsensusADMMLasso(lambda_reg=lambda_reg, rho=rho, max_iter=1000, tol=1e-6)
    
    # 添加節點
    for i, (A_i, b_i) in enumerate(zip(A_splits, b_splits)):
        consensus_admm.add_node(A_i, b_i)
    
    x_consensus = consensus_admm.fit(A_full=A, b_full=b, x_true=x_true, verbose=True)
    
    # 性能比較
    print(f"\n6. 性能比較:")
    
    # 權重估計誤差
    mse_single = np.mean((x_single - x_true)**2)
    mse_consensus = np.mean((x_consensus - x_true)**2)
    correlation_single = np.corrcoef(x_true, x_single)[0, 1]
    correlation_consensus = np.corrcoef(x_true, x_consensus)[0, 1]
    
    print(f"   權重估計 MSE:")
    print(f"     單一 ADMM: {mse_single:.8f}")
    print(f"     Consensus ADMM: {mse_consensus:.8f}")
    print(f"     相對差異: {abs(mse_single - mse_consensus)/mse_single*100:.4f}%")
    
    print(f"   權重相關係數:")
    print(f"     單一 ADMM: {correlation_single:.8f}")
    print(f"     Consensus ADMM: {correlation_consensus:.8f}")
    
    # 預測性能
    r2_single = 1 - np.sum((b - A @ x_single)**2) / np.sum((b - np.mean(b))**2)
    r2_consensus = 1 - np.sum((b - A @ x_consensus)**2) / np.sum((b - np.mean(b))**2)
    
    print(f"   預測 R²:")
    print(f"     單一 ADMM: {r2_single:.8f}")
    print(f"     Consensus ADMM: {r2_consensus:.8f}")
    
    # 收斂性能
    print(f"   收斂性能:")
    print(f"     單一 ADMM 迭代數: {len(single_admm.objective_history)}")
    print(f"     Consensus ADMM 迭代數: {len(consensus_admm.objective_history)}")
    print(f"     單一 ADMM 時間: {single_admm.total_time:.4f} 秒")
    print(f"     Consensus ADMM 時間: {consensus_admm.total_time:.4f} 秒")
    print(f"     通信輪數: {consensus_admm.communication_rounds}")
    
    # 權重一致性檢查
    weight_diff_norm = np.linalg.norm(x_single - x_consensus)
    weight_diff_max = np.max(np.abs(x_single - x_consensus))
    
    print(f"\n7. 一致性分析:")
    print(f"   權重差異 L2 範数: {weight_diff_norm:.8f}")
    print(f"   權重差異最大值: {weight_diff_max:.8f}")
    print(f"   權重相關係數: {np.corrcoef(x_single, x_consensus)[0,1]:.8f}")
    
    # 目標函數值比較
    final_obj_single = single_admm.objective_history[-1]
    final_obj_consensus = consensus_admm.objective_history[-1]
    
    print(f"   最終目標函數值:")
    print(f"     單一 ADMM: {final_obj_single:.8f}")
    print(f"     Consensus ADMM: {final_obj_consensus:.8f}")
    print(f"     相對差異: {abs(final_obj_single - final_obj_consensus)/final_obj_single*100:.6f}%")
    
    # 繪製比較圖
    print(f"\n8. 繪製比較結果...")
    plot_consensus_comparison(single_admm, consensus_admm, x_true, x_single, x_consensus, A, b)
    
    # 結論
    print(f"\n✅ Consensus ADMM vs 單一 ADMM 比較完成!")
    if weight_diff_norm < 1e-4:
        print("   ✅ 兩種方法得到了高度一致的結果!")
    elif weight_diff_norm < 1e-2:
        print("   ⚠️  兩種方法結果基本一致，存在輕微差異")
    else:
        print("   ❌ 兩種方法結果存在明顯差異，需要檢查實現")
    
    print(f"   權重估計一致性: {weight_diff_norm:.2e}")
    print(f"   目標函數一致性: {abs(final_obj_single - final_obj_consensus):.2e}")

if __name__ == "__main__":
    main()