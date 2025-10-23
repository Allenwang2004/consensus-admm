import numpy as np

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