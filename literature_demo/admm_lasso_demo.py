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

# 設置中文字體支持
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

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

def generate_sparse_data(m=100, n=50, sparsity=0.1, noise_std=0.1, random_state=42):
    """
    生成稀疏回歸數據
    
    參數:
    m: 樣本數
    n: 特徵數
    sparsity: 稀疏度 (非零係數的比例)
    noise_std: 噪聲標準差
    random_state: 隨機種子
    """
    np.random.seed(random_state)
    
    # 生成設計矩陣 A
    A = np.random.randn(m, n)
    A = A / np.sqrt(np.sum(A**2, axis=0))  # 標準化列
    
    # 生成稀疏真實權重
    x_true = np.zeros(n)
    n_nonzero = int(sparsity * n)
    nonzero_indices = np.random.choice(n, n_nonzero, replace=False)
    x_true[nonzero_indices] = np.random.randn(n_nonzero) * 2  # 較大的係數
    
    # 生成目標向量
    b = A @ x_true + noise_std * np.random.randn(m)
    
    return A, b, x_true

def plot_results(admm_solver, A, b, x_true, x_admm):
    """
    繪製結果圖表
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 目標函數收斂曲線
    axes[0, 0].semilogy(admm_solver.objective_history)
    axes[0, 0].set_title('目標函數收斂曲線')
    axes[0, 0].set_xlabel('迭代次數')
    axes[0, 0].set_ylabel('目標函數值 (log scale)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 殘差收斂曲線
    primal_res = [r[0] for r in admm_solver.residual_history]
    dual_res = [r[1] for r in admm_solver.residual_history]
    axes[0, 1].semilogy(primal_res, label='原始殘差', alpha=0.8)
    axes[0, 1].semilogy(dual_res, label='對偶殘差', alpha=0.8)
    axes[0, 1].set_title('殘差收斂曲線')
    axes[0, 1].set_xlabel('迭代次數')
    axes[0, 1].set_ylabel('殘差 (log scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 真實權重 vs 估計權重
    indices = np.arange(len(x_true))
    width = 0.35
    axes[1, 0].bar(indices - width/2, x_true, width, label='真實權重', alpha=0.7)
    axes[1, 0].bar(indices + width/2, x_admm, width, label='ADMM 估計', alpha=0.7)
    axes[1, 0].set_title('權重分佈比較')
    axes[1, 0].set_xlabel('特徵索引')
    axes[1, 0].set_ylabel('權重值')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 散點圖: 真實 vs 估計
    axes[1, 1].scatter(x_true, x_admm, alpha=0.7)
    min_val = min(np.min(x_true), np.min(x_admm))
    max_val = max(np.max(x_true), np.max(x_admm))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 1].set_title('真實權重 vs 估計權重')
    axes[1, 1].set_xlabel('真實權重')
    axes[1, 1].set_ylabel('估計權重')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 預測值比較: Ax vs 真實 b
    b_pred_true = A @ x_true
    b_pred_admm = A @ x_admm
    
    axes[0, 2].scatter(b, b_pred_true, alpha=0.7, label='真實權重預測', color='blue')
    axes[0, 2].scatter(b, b_pred_admm, alpha=0.7, label='ADMM 權重預測', color='red')
    min_b = min(np.min(b), np.min(b_pred_true), np.min(b_pred_admm))
    max_b = max(np.max(b), np.max(b_pred_true), np.max(b_pred_admm))
    axes[0, 2].plot([min_b, max_b], [min_b, max_b], 'k--', alpha=0.8, label='理想線')
    axes[0, 2].set_title('預測值 vs 真實值')
    axes[0, 2].set_xlabel('真實 b')
    axes[0, 2].set_ylabel('預測 Ax')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 6. 預測誤差分析
    error_true = b - b_pred_true
    error_admm = b - b_pred_admm
    
    axes[1, 2].hist(error_true, bins=20, alpha=0.7, label='真實權重誤差', color='blue', density=True)
    axes[1, 2].hist(error_admm, bins=20, alpha=0.7, label='ADMM 權重誤差', color='red', density=True)
    axes[1, 2].set_title('預測誤差分佈')
    axes[1, 2].set_xlabel('預測誤差 (b - Ax)')
    axes[1, 2].set_ylabel('密度')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函數: 演示 ADMM Lasso 算法
    """
    print("=" * 60)
    print("ADMM 算法解決 Lasso 回歸問題演示")
    print("=" * 60)
    
    # 生成合成數據
    print("\n生成合成稀疏數據...")
    m, n = 100, 50  # 100 樣本, 50 特徵
    sparsity = 0.2  # 20% 非零係數
    A, b, x_true = generate_sparse_data(m, n, sparsity=sparsity, noise_std=0.1)
    
    # print(f"   數據維度: {A.shape}")
    # print(f"   稀疏度: {sparsity}")
    # print(f"   真實非零係數數量: {np.sum(x_true != 0)}")
    # print(f"   最大真實權重: {np.max(np.abs(x_true)):.3f}")
    
    # 設置 ADMM 參數
    lambda_reg = 0.1  # L1 正則化參數
    rho = 1.0         # 增廣拉格朗日乘數
    
    # print(f"\n2. ADMM 參數設置:")
    # print(f"   λ (正則化參數) = {lambda_reg}")
    # print(f"   ρ (增廣參數) = {rho}")
    
    # 初始化並訓練 ADMM 求解器
    print(f"\n開始 ADMM 求解...")
    admm_solver = ADMMLasso(lambda_reg=lambda_reg, rho=rho, max_iter=1000, tol=1e-8)
    x_admm = admm_solver.fit(A, b, x_true=x_true, verbose=True)
    
    # 計算性能指標
    print(f"\n性能評估:")
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
    
    # print(f"   權重估計指標:")
    # print(f"     均方誤差 (MSE): {mse:.6f}")
    # print(f"     平均絕對誤差 (MAE): {mae:.6f}")
    # print(f"     相關係數: {correlation:.6f}")
    # print(f"     真實非零係數: {n_nonzero_true}")
    # print(f"     估計非零係數: {n_nonzero_admm}")
    
    print(f"   預測性能指標:")
    print(f"     真實權重 R²: {r2_true:.6f}")
    print(f"     ADMM 權重 R²: {r2_admm:.6f}")
    print(f"     真實權重 RMSE: {np.sqrt(np.mean((b - b_pred_true)**2)):.6f}")
    print(f"     ADMM 權重 RMSE: {np.sqrt(np.mean((b - b_pred_admm)**2)):.6f}")
    # print(f"     預測相關係數 (真實): {np.corrcoef(b, b_pred_true)[0,1]:.6f}")
    # print(f"     預測相關係數 (ADMM): {np.corrcoef(b, b_pred_admm)[0,1]:.6f}")
    
    # 目標函數下降驗證
    # print(f"\n5. 收斂性驗證:")
    # obj_initial = admm_solver.objective_history[0]
    # obj_final = admm_solver.objective_history[-1]
    # obj_reduction = (obj_initial - obj_final) / obj_initial * 100
    
    # print(f"   初始目標函數值: {obj_initial:.6f}")
    # print(f"   最終目標函數值: {obj_final:.6f}")
    # print(f"   目標函數下降: {obj_reduction:.2f}%")
    # print(f"   總迭代次數: {len(admm_solver.objective_history)}")
    
    # # 檢查是否單調下降
    # is_decreasing = all(admm_solver.objective_history[i] >= admm_solver.objective_history[i+1] 
    #                    for i in range(len(admm_solver.objective_history)-1))
    # print(f"   目標函數單調下降: {'是' if is_decreasing else '否'}")
    
    # 繪製結果
    # print(f"\n6. 繪製結果圖表...")
    # plot_results(admm_solver, A, b, x_true, x_admm)
    
    # 顯示權重對比的詳細信息
    # print(f"\n7. 權重分佈詳細比較:")
    # print("   索引  真實權重   ADMM估計    誤差")
    # print("   " + "-" * 35)
    # for i in range(min(15, len(x_true))):  # 只顯示前15個
    #     error = abs(x_true[i] - x_admm[i])
    #     print(f"   {i:2d}   {x_true[i]:8.4f}  {x_admm[i]:8.4f}  {error:.4f}")
    # if len(x_true) > 15:
    #     print("   ...")
    
    # # 顯示預測值對比的詳細信息
    # print(f"\n8. 預測值 (Ax) 與真實值 (b) 比較:")
    # b_pred_true = A @ x_true
    # b_pred_admm = A @ x_admm
    # print("   樣本  真實b      Ax(真實)   Ax(ADMM)   誤差(真實)  誤差(ADMM)")
    # print("   " + "-" * 65)
    # for i in range(min(10, len(b))):  # 只顯示前10個樣本
    #     error_true = abs(b[i] - b_pred_true[i])
    #     error_admm = abs(b[i] - b_pred_admm[i])
    #     print(f"   {i:2d}   {b[i]:8.4f}  {b_pred_true[i]:8.4f}  {b_pred_admm[i]:8.4f}  "
    #           f"{error_true:8.4f}  {error_admm:8.4f}")
    # if len(b) > 10:
    #     print("   ...")
    
    # # 統計摘要
    # print(f"\n9. 預測誤差統計摘要:")
    # error_true_stats = b - b_pred_true
    # error_admm_stats = b - b_pred_admm
    # print(f"   真實權重預測誤差:")
    # print(f"     均值: {np.mean(error_true_stats):8.4f}")
    # print(f"     標準差: {np.std(error_true_stats):8.4f}")
    # print(f"     最大絕對誤差: {np.max(np.abs(error_true_stats)):8.4f}")
    # print(f"   ADMM 權重預測誤差:")
    # print(f"     均值: {np.mean(error_admm_stats):8.4f}")
    # print(f"     標準差: {np.std(error_admm_stats):8.4f}")
    # print(f"     最大絕對誤差: {np.max(np.abs(error_admm_stats)):8.4f}")
    
    # print(f"\nADMM Lasso 演示完成!")
    # print(f"   目標函數成功下降 {obj_reduction:.2f}%")
    # print(f"   權重估計相關係數: {correlation:.4f}")
    # print(f"   ADMM 預測 R²: {r2_admm:.4f}")
    # print(f"   相對於真實權重的預測性能損失: {((r2_true - r2_admm)/r2_true*100):.2f}%")

if __name__ == "__main__":
    main()