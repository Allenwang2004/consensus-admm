#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <Eigen/Dense>

namespace consensus_admm {

/**
 * @brief Parameters for the Consensus ADMM algorithm
 */
struct ADMMParams {
    double rho = 1.0;           // penalty parameter
    int max_iters = 1000;       // maximum number of iterations
    double tol = 1e-4;          // convergence tolerance
    bool verbose = false;       // print convergence information
};

/**
 * @brief Results from Consensus ADMM optimization
 */
struct ADMMResults {
    Eigen::VectorXd consensus;                      // consensus variable z
    std::vector<Eigen::VectorXd> local_variables;   // local variables x_i
    std::vector<Eigen::VectorXd> dual_variables;    // dual variables u_i
    std::vector<double> primal_residuals;           // convergence history
    std::vector<double> dual_residuals;             // convergence history
    int iterations;                                 // number of iterations
    bool converged;                                 // convergence flag
};

/**
 * @brief Type definitions for user-provided functions
 */
using LossFunction = std::function<double(const Eigen::VectorXd&)>;
using GradientFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
using ProximalOperator = std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>;

/**
 * @brief Consensus ADMM solver class
 */
class ConsensusADMM {
public:
    /**
     * @brief Constructor
     * @param num_agents Number of agents in the consensus problem
     * @param variable_dim Dimension of variables
     * @param params ADMM parameters
     */
    ConsensusADMM(int num_agents, int variable_dim, const ADMMParams& params = ADMMParams{});

    /**
     * @brief Set loss function and gradient for agent i
     * @param agent_id Agent index (0-based)
     * @param loss_func Loss function f_i(x)
     * @param grad_func Gradient function ∇f_i(x)
     */
    void set_agent_loss(int agent_id, const LossFunction& loss_func, 
                       const GradientFunction& grad_func);

    /**
     * @brief Set proximal operator for agent i (for non-smooth terms)
     * @param agent_id Agent index (0-based)
     * @param prox_func Proximal operator prox_{g_i}(x, ρ)
     */
    void set_agent_proximal(int agent_id, const ProximalOperator& prox_func);

    /**
     * @brief Initialize variables with given values
     * @param initial_x Initial local variables
     * @param initial_z Initial consensus variable
     */
    void initialize(const std::vector<Eigen::VectorXd>& initial_x,
                   const Eigen::VectorXd& initial_z);

    /**
     * @brief Solve the consensus ADMM problem
     * @return Results containing optimal variables and convergence info
     */
    ADMMResults solve();

    /**
     * @brief Get current parameters
     */
    const ADMMParams& get_params() const { return params_; }

    /**
     * @brief Update parameters
     */
    void set_params(const ADMMParams& params) { params_ = params; }

private:
    int num_agents_;
    int variable_dim_;
    ADMMParams params_;

    // Variables
    std::vector<Eigen::VectorXd> x_;    // local variables
    Eigen::VectorXd z_;                 // consensus variable
    std::vector<Eigen::VectorXd> u_;    // dual variables

    // User-provided functions
    std::vector<LossFunction> loss_functions_;
    std::vector<GradientFunction> gradient_functions_;
    std::vector<ProximalOperator> proximal_operators_;

    // Private methods
    void update_local_variables();
    void update_consensus_variable();
    void update_dual_variables();
    double compute_primal_residual();
    double compute_dual_residual();
    bool check_convergence(double primal_res, double dual_res);
};

} // namespace consensus_admm