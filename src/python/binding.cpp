#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "consensus_admm.hpp"

namespace py = pybind11;
using namespace consensus_admm;

PYBIND11_MODULE(consensus_admm_cpp, m) {
    m.doc() = "Consensus ADMM optimization toolkit";
    
    // Bind ADMMParams struct
    py::class_<ADMMParams>(m, "ADMMParams")
        .def(py::init<>())
        .def_readwrite("rho", &ADMMParams::rho)
        .def_readwrite("max_iters", &ADMMParams::max_iters)
        .def_readwrite("tol", &ADMMParams::tol)
        .def_readwrite("verbose", &ADMMParams::verbose);
    
    // Bind ADMMResults struct
    py::class_<ADMMResults>(m, "ADMMResults")
        .def(py::init<>())
        .def_readonly("consensus", &ADMMResults::consensus)
        .def_readonly("local_variables", &ADMMResults::local_variables)
        .def_readonly("dual_variables", &ADMMResults::dual_variables)
        .def_readonly("primal_residuals", &ADMMResults::primal_residuals)
        .def_readonly("dual_residuals", &ADMMResults::dual_residuals)
        .def_readonly("iterations", &ADMMResults::iterations)
        .def_readonly("converged", &ADMMResults::converged);
    
    // Bind ConsensusADMM class
    py::class_<ConsensusADMM>(m, "ConsensusADMM")
        .def(py::init<int, int, const ADMMParams&>(),
             py::arg("num_agents"), 
             py::arg("variable_dim"), 
             py::arg("params") = ADMMParams{})
        .def("set_agent_loss", &ConsensusADMM::set_agent_loss,
             "Set loss function and gradient for an agent",
             py::arg("agent_id"), py::arg("loss_func"), py::arg("grad_func"))
        .def("set_agent_proximal", &ConsensusADMM::set_agent_proximal,
             "Set proximal operator for an agent",
             py::arg("agent_id"), py::arg("prox_func"))
        .def("initialize", &ConsensusADMM::initialize,
             "Initialize variables",
             py::arg("initial_x"), py::arg("initial_z"))
        .def("solve", &ConsensusADMM::solve,
             "Solve the consensus ADMM problem")
        .def("get_params", &ConsensusADMM::get_params,
             "Get current parameters",
             py::return_value_policy::reference_internal)
        .def("set_params", &ConsensusADMM::set_params,
             "Update parameters",
             py::arg("params"));
    
    // Helper functions for common proximal operators
    m.def("soft_threshold", [](const Eigen::VectorXd& x, double lambda) {
        Eigen::VectorXd result(x.size());
        for (int i = 0; i < x.size(); ++i) {
            if (x[i] > lambda) {
                result[i] = x[i] - lambda;
            } else if (x[i] < -lambda) {
                result[i] = x[i] + lambda;
            } else {
                result[i] = 0.0;
            }
        }
        return result;
    }, "Soft thresholding operator for L1 regularization");
    
    m.def("l2_proximal", [](const Eigen::VectorXd& x, double lambda) {
        return x / (1.0 + lambda);
    }, "Proximal operator for L2 regularization");
}