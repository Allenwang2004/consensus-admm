"""
ADMM 算法解決 Lasso 回歸問題
使用交替方向乘數法 (Alternating Direction Method of Multipliers)

Lasso 問題: minimize (1/2)||Ax - b||^2 + λ||x||_1

ADMM 分解:
minimize (1/2)||Ax - b||^2 + λ||z||_1
subject to x - z = 0

擴展拉格朗日函數:
L(x,z,u) = (1/2)||Ax - b||^2 + λ||z||_1 + u^T(x - z) + (ρ/2)||x - z||^2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
from utils import generate_sparse_data

class ADMMLasso:
    def __init__(self, lambda_reg=0.1, rho=1.0, max_iter=1000, tol=1e-6):
        """
        ADMM Lasso 求解器
        
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
        self.objective_history = []
        self.residual_history = []
        
    def soft_threshold(self, x, threshold):
        """
        軟閾值函數 (用於 L1 近端算子)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def compute_objective(self, A, b, x, z):
        """
        計算 Lasso 目標函數值
        """
        return 0.5 * np.linalg.norm(A @ x - b)**2 + self.lambda_reg * np.linalg.norm(z, 1)
    
    def fit(self, A, b, x_true=None, verbose=False):
        """
        使用 ADMM 算法求解 Lasso 問題
        
        參數:
        A: 設計矩陣 (m x n)
        b: 目標向量 (m,)
        x_true: 真實權重 (用於監控誤差)
        verbose: 是否打印迭代信息
        """
        m, n = A.shape
        
        # 初始化變量
        x = np.zeros(n)  # 主變量
        z = np.zeros(n)  # 輔助變量
        u = np.zeros(n)  # 對偶變量
        
        # 預計算 A^T A 和 A^T b (用於 x 更新)
        AtA = A.T @ A
        Atb = A.T @ b
        AtA_rhoI_inv = np.linalg.inv(AtA + self.rho * np.eye(n))
        
        # 清空歷史記錄
        self.objective_history = []
        self.residual_history = []
        
        if verbose:
            print("開始 ADMM 迭代...")
            print(f"λ = {self.lambda_reg}, ρ = {self.rho}")
            print("-" * 60)
        
        for k in range(self.max_iter):
            x_old = x.copy()
            z_old = z.copy()
            
            # 步驟 1: 更新 x (二次規劃)
            # x^{k+1} = argmin_x (1/2)||Ax - b||^2 + (ρ/2)||x - z^k + u^k||^2
            x = AtA_rhoI_inv @ (Atb + self.rho * (z - u))
            
            # 步驟 2: 更新 z (軟閾值)
            # z^{k+1} = argmin_z λ||z||_1 + (ρ/2)||x^{k+1} - z + u^k||^2
            z = self.soft_threshold(x + u, self.lambda_reg / self.rho)
            
            # 步驟 3: 更新對偶變量 u
            # u^{k+1} = u^k + x^{k+1} - z^{k+1}
            u = u + x - z
            
            # 計算目標函數值
            obj_val = self.compute_objective(A, b, x, z)
            self.objective_history.append(obj_val)
            
            # 計算原始和對偶殘差
            primal_residual = np.linalg.norm(x - z)
            dual_residual = self.rho * np.linalg.norm(z - z_old)
            self.residual_history.append((primal_residual, dual_residual))
            
            # 檢查收斂
            if primal_residual < self.tol and dual_residual < self.tol:
                if verbose:
                    print(f"迭代 {k+1}: 收斂!")
                break
            
            # 打印進度
            if verbose and (k+1) % 50 == 0:
                print(f"迭代 {k+1:4d}: 目標函數 = {obj_val:.6f}, "
                      f"原始殘差 = {primal_residual:.2e}, "
                      f"對偶殘差 = {dual_residual:.2e}")
                if x_true is not None:
                    mse = np.mean((x - x_true)**2)
                    print(f"           MSE (vs 真實) = {mse:.6f}")
        
        if verbose:
            print("-" * 60)
            print(f"ADMM 完成! 總迭代次數: {k+1}")
            print(f"最終目標函數值: {self.objective_history[-1]:.6f}")
        
        self.x_final = x
        self.z_final = z
        return x
    
if __name__ == "__main__":
    m, n = 1000000, 2000  # 100 樣本, 50 特徵
    sparsity = 0.2  # 20% 非零係數
    A, b, x_true = generate_sparse_data(m, n, sparsity=sparsity, noise_std=0.1)

    lambda_reg = 0.1  # L1 正則化參數
    rho = 1.0         # 增廣拉格朗日乘數

    admm_solver = ADMMLasso(lambda_reg=lambda_reg, rho=rho, max_iter=1000, tol=1e-8)
    x_admm = admm_solver.fit(A, b, x_true=x_true, verbose=True)

    mse = np.mean((x_admm - x_true)**2)
    mae = np.mean(np.abs(x_admm - x_true))
    correlation = np.corrcoef(x_true, x_admm)[0, 1]
    
    # 稀疏性比較
    n_nonzero_true = np.sum(np.abs(x_true) > 1e-6)
    n_nonzero_admm = np.sum(np.abs(x_admm) > 1e-6)
    
    # 預測性能評估
    b_pred_true = A @ x_true
    b_pred_admm = A @ x_admm
    
    # 計算 R² 分數
    ss_res_true = np.sum((b - b_pred_true) ** 2)
    ss_res_admm = np.sum((b - b_pred_admm) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    r2_true = 1 - (ss_res_true / ss_tot)
    r2_admm = 1 - (ss_res_admm / ss_tot)

    print(f"   預測性能指標:")
    print(f"     真實權重 R²: {r2_true:.6f}")
    print(f"     ADMM 權重 R²: {r2_admm:.6f}")
    print(f"     真實權重 RMSE: {np.sqrt(np.mean((b - b_pred_true)**2)):.6f}")
    print(f"     ADMM 權重 RMSE: {np.sqrt(np.mean((b - b_pred_admm)**2)):.6f}")