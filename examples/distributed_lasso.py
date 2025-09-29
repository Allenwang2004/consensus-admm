"""
Simple example demonstrating distributed Lasso regression using Consensus ADMM.
"""

import numpy as np
import matplotlib.pyplot as plt
from consensus_admm import ConsensusADMM, soft_threshold

def distributed_lasso_example():
    """Solve distributed Lasso: min Σ(0.5||A_i*x - b_i||² + λ||x||₁)"""
    
    # Problem setup
    num_agents = 5
    variable_dim = 20
    samples_per_agent = 50
    lambda_reg = 0.1
    noise_level = 0.1
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Create sparse true solution
    true_x = np.random.randn(variable_dim)
    true_x[np.abs(true_x) < 0.5] = 0  # Make it sparse
    
    # Generate data for each agent
    data = []
    for i in range(num_agents):
        A_i = np.random.randn(samples_per_agent, variable_dim)
        b_i = A_i @ true_x + noise_level * np.random.randn(samples_per_agent)
        data.append((A_i, b_i))
    
    print(f"Generated synthetic data:")
    print(f"  - Number of agents: {num_agents}")
    print(f"  - Variable dimension: {variable_dim}")
    print(f"  - Samples per agent: {samples_per_agent}")
    print(f"  - True sparsity: {np.sum(true_x != 0)}/{variable_dim}")
    
    # Define loss functions (least squares part)
    def make_ls_loss(A, b):
        def loss(x):
            residual = A @ x - b
            return 0.5 * np.dot(residual, residual)
        return loss
    
    def make_ls_grad(A, b):
        def grad(x):
            return A.T @ (A @ x - b)
        return grad
    
    # L1 proximal operator 
    def l1_prox(x, rho):
        return soft_threshold(x, lambda_reg / rho)
    
    # Set up solver
    loss_functions = [make_ls_loss(A, b) for A, b in data]
    grad_functions = [make_ls_grad(A, b) for A, b in data]
    prox_functions = [l1_prox for _ in range(num_agents)]
    
    solver = ConsensusADMM(
        num_agents=num_agents,
        local_losses=loss_functions,
        local_gradients=grad_functions,
        local_prox_ops=prox_functions,
        rho=1.0,
        max_iters=500,
        tol=1e-6,
        verbose=True
    )
    
    # Solve
    print("\nSolving distributed Lasso with Consensus ADMM...")
    initial_x = [np.zeros(variable_dim) for _ in range(num_agents)]
    initial_z = np.zeros(variable_dim)
    
    results = solver.solve(initial_x=initial_x, initial_z=initial_z)
    
    # Extract solution
    x_admm = results['consensus']
    
    print(f"\nResults:")
    print(f"  - Converged: {results['converged']}")
    print(f"  - Iterations: {results['iterations']}")
    print(f"  - Final primal residual: {results['primal_residuals'][-1]:.2e}")
    print(f"  - Solution sparsity: {np.sum(np.abs(x_admm) > 1e-3)}/{variable_dim}")
    
    # Compare with true solution
    mse = np.mean((x_admm - true_x)**2)
    print(f"  - MSE vs true solution: {mse:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.semilogy(results['primal_residuals'])
    plt.xlabel('Iteration')
    plt.ylabel('Primal Residual')
    plt.title('Convergence')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.stem(range(variable_dim), true_x, basefmt=' ', label='True', alpha=0.7)
    plt.stem(range(variable_dim), x_admm, basefmt=' ', label='ADMM', alpha=0.7)
    plt.xlabel('Variable Index')
    plt.ylabel('Value')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.scatter(true_x, x_admm, alpha=0.7)
    plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
    plt.xlabel('True Value')
    plt.ylabel('ADMM Value')
    plt.title('True vs Recovered')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lasso_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    distributed_lasso_example()