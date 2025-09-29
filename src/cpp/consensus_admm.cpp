#include "consensus_admm.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace consensus_admm {

ConsensusADMM::ConsensusADMM(int num_agents, int variable_dim, const ADMMParams& params)
    : num_agents_(num_agents), variable_dim_(variable_dim), params_(params) {
    
    // Initialize variables
    x_.resize(num_agents_);
    u_.resize(num_agents_);
    z_ = Eigen::VectorXd::Zero(variable_dim_);
    
    for (int i = 0; i < num_agents_; ++i) {
        x_[i] = Eigen::VectorXd::Zero(variable_dim_);
        u_[i] = Eigen::VectorXd::Zero(variable_dim_);
    }
    
    // Initialize function vectors
    loss_functions_.resize(num_agents_);
    gradient_functions_.resize(num_agents_);
    proximal_operators_.resize(num_agents_);
}

void ConsensusADMM::set_agent_loss(int agent_id, const LossFunction& loss_func, 
                                  const GradientFunction& grad_func) {
    if (agent_id < 0 || agent_id >= num_agents_) {
        throw std::invalid_argument("Invalid agent ID");
    }
    loss_functions_[agent_id] = loss_func;
    gradient_functions_[agent_id] = grad_func;
}

void ConsensusADMM::set_agent_proximal(int agent_id, const ProximalOperator& prox_func) {
    if (agent_id < 0 || agent_id >= num_agents_) {
        throw std::invalid_argument("Invalid agent ID");
    }
    proximal_operators_[agent_id] = prox_func;
}

void ConsensusADMM::initialize(const std::vector<Eigen::VectorXd>& initial_x,
                              const Eigen::VectorXd& initial_z) {
    if (initial_x.size() != num_agents_) {
        throw std::invalid_argument("Number of initial x vectors must match number of agents");
    }
    
    for (int i = 0; i < num_agents_; ++i) {
        if (initial_x[i].size() != variable_dim_) {
            throw std::invalid_argument("Initial x dimensions must match variable dimension");
        }
        x_[i] = initial_x[i];
    }
    
    if (initial_z.size() != variable_dim_) {
        throw std::invalid_argument("Initial z dimension must match variable dimension");
    }
    z_ = initial_z;
}

ADMMResults ConsensusADMM::solve() {
    ADMMResults results;
    results.primal_residuals.clear();
    results.dual_residuals.clear();
    
    if (params_.verbose) {
        std::cout << "Starting Consensus ADMM optimization..." << std::endl;
        std::cout << "Agents: " << num_agents_ << ", Dimension: " << variable_dim_ << std::endl;
    }
    
    for (int iter = 0; iter < params_.max_iters; ++iter) {
        // Store previous z for dual residual calculation
        Eigen::VectorXd z_old = z_;
        
        // ADMM updates
        update_local_variables();
        update_consensus_variable();
        update_dual_variables();
        
        // Compute residuals
        double primal_res = compute_primal_residual();
        double dual_res = compute_dual_residual();
        
        results.primal_residuals.push_back(primal_res);
        results.dual_residuals.push_back(dual_res);
        
        if (params_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter << ": primal_res = " << primal_res 
                     << ", dual_res = " << dual_res << std::endl;
        }
        
        // Check convergence
        if (check_convergence(primal_res, dual_res)) {
            results.converged = true;
            results.iterations = iter + 1;
            if (params_.verbose) {
                std::cout << "Converged in " << results.iterations << " iterations" << std::endl;
            }
            break;
        }
        
        results.iterations = iter + 1;
    }
    
    if (!results.converged && params_.verbose) {
        std::cout << "Maximum iterations reached without convergence" << std::endl;
    }
    
    // Store final results
    results.consensus = z_;
    results.local_variables = x_;
    results.dual_variables = u_;
    
    return results;
}

void ConsensusADMM::update_local_variables() {
    for (int i = 0; i < num_agents_; ++i) {
        if (gradient_functions_[i] && proximal_operators_[i]) {
            // For problems with both smooth and non-smooth terms
            // Use gradient step + proximal operator
            Eigen::VectorXd grad = gradient_functions_[i](x_[i]);
            Eigen::VectorXd temp = x_[i] - (1.0 / params_.rho) * grad + z_ - u_[i];
            x_[i] = proximal_operators_[i](temp, 1.0 / params_.rho);
        } else if (gradient_functions_[i]) {
            // For smooth problems: solve x_i = argmin { f_i(x) + (ρ/2)||x - z + u_i||² }
            // Using gradient descent (can be replaced with more sophisticated methods)
            Eigen::VectorXd grad = gradient_functions_[i](x_[i]);
            x_[i] = x_[i] - 0.01 * (grad + params_.rho * (x_[i] - z_ + u_[i]));
        } else if (proximal_operators_[i]) {
            // For non-smooth problems only
            x_[i] = proximal_operators_[i](z_ - u_[i], 1.0 / params_.rho);
        }
    }
}

void ConsensusADMM::update_consensus_variable() {
    // z^{k+1} = (1/N) * Σ(x_i^{k+1} + u_i^k)
    z_.setZero();
    for (int i = 0; i < num_agents_; ++i) {
        z_ += x_[i] + u_[i];
    }
    z_ /= static_cast<double>(num_agents_);
}

void ConsensusADMM::update_dual_variables() {
    // u_i^{k+1} = u_i^k + (x_i^{k+1} - z^{k+1})
    for (int i = 0; i < num_agents_; ++i) {
        u_[i] += (x_[i] - z_);
    }
}

double ConsensusADMM::compute_primal_residual() {
    // ||r^k||₂ where r^k = [x_1^k - z^k; ...; x_N^k - z^k]
    double res = 0.0;
    for (int i = 0; i < num_agents_; ++i) {
        res += (x_[i] - z_).squaredNorm();
    }
    return std::sqrt(res);
}

double ConsensusADMM::compute_dual_residual() {
    // This is a simplified dual residual computation
    // In practice, this depends on the specific problem structure
    return 0.0; // Placeholder - implement based on specific problem
}

bool ConsensusADMM::check_convergence(double primal_res, double dual_res) {
    // Simple convergence check based on primal residual
    return primal_res < params_.tol;
}

} // namespace consensus_admm