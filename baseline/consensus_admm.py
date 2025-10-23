"""
Consensus ADMM 算法解決 Lasso 回歸問題
將數據分散到多個節點，每個節點運行局部 ADMM 並通過全域變量協調

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
from utils import generate_sparse_data, split_data
warnings.filterwarnings('ignore')

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
    
if __name__ == "__main__":
    m, n = 200, 50  # 200 樣本, 50 特徵
    sparsity = 0.2
    A, b, x_true = generate_sparse_data(m, n, sparsity=sparsity, noise_std=0.1)

    lambda_reg = 0.1
    rho = 1.0
    num_nodes = 4  # Consensus ADMM 節點數

    A_splits, b_splits = split_data(A, b, num_nodes, method='row')

    consensus_admm = ConsensusADMMLasso(lambda_reg=lambda_reg, rho=rho, max_iter=1000, tol=1e-6)

    for i, (A_i, b_i) in enumerate(zip(A_splits, b_splits)):
        consensus_admm.add_node(A_i, b_i)
    
    x_consensus = consensus_admm.fit(A_full=A, b_full=b, x_true=x_true, verbose=True)
    

    mse = np.mean((x_consensus - x_true)**2)
    mae = np.mean(np.abs(x_consensus - x_true))
    correlation = np.corrcoef(x_true, x_consensus)[0, 1]

    r2_consensus = 1 - np.sum((b - A @ x_consensus)**2) / np.sum((b - np.mean(b))**2)

    print(f"\nConsensus ADMM 結果:")
    print(f"權重估計 MSE: {mse:.8f}")
    print(f"權重估計 MAE: {mae:.8f}")
    print(f"權重相關係數: {correlation:.8f}")
    print(f"預測 R²: {r2_consensus:.8f}")