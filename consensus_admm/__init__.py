"""
Python interface for Consensus ADMM optimization toolkit.
"""

import numpy as np
from typing import List, Callable, Dict, Optional, Union
from .consensus_admm_cpp import ConsensusADMM as _ConsensusADMM
from .consensus_admm_cpp import ADMMParams, ADMMResults
from .consensus_admm_cpp import soft_threshold, l2_proximal

__all__ = ['ConsensusADMM', 'ADMMParams', 'ADMMResults', 'soft_threshold', 'l2_proximal']


class ConsensusADMM:
    """
    High-level Python interface for Consensus ADMM optimization.
    
    This class provides a convenient Python API that wraps the C++ implementation
    and handles the conversion between Python functions and C++ callbacks.
    """
    
    def __init__(
        self, 
        num_agents: int,
        local_losses: Optional[List[Callable[[np.ndarray], float]]] = None,
        local_gradients: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        local_prox_ops: Optional[List[Callable[[np.ndarray, float], np.ndarray]]] = None,
        rho: float = 1.0,
        max_iters: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        """
        Initialize Consensus ADMM solver.
        
        Parameters:
        -----------
        num_agents : int
            Number of agents in the consensus problem
        local_losses : list of callable, optional
            Loss functions f_i(x) for each agent
        local_gradients : list of callable, optional  
            Gradient functions ∇f_i(x) for each agent
        local_prox_ops : list of callable, optional
            Proximal operators prox_{g_i}(x, ρ) for each agent
        rho : float, default=1.0
            Penalty parameter
        max_iters : int, default=1000
            Maximum number of iterations
        tol : float, default=1e-4
            Convergence tolerance
        verbose : bool, default=False
            Print convergence information
        """
        self.num_agents = num_agents
        self.variable_dim = None  # Will be set when solve() is called
        
        # Store Python functions
        self.local_losses = local_losses or [None] * num_agents
        self.local_gradients = local_gradients or [None] * num_agents  
        self.local_prox_ops = local_prox_ops or [None] * num_agents
        
        # Create parameters
        self.params = ADMMParams()
        self.params.rho = rho
        self.params.max_iters = max_iters
        self.params.tol = tol
        self.params.verbose = verbose
        
        self._cpp_solver = None
    
    def _initialize_cpp_solver(self, variable_dim: int):
        """Initialize the C++ solver with the given variable dimension."""
        if self._cpp_solver is None or self.variable_dim != variable_dim:
            self.variable_dim = variable_dim
            self._cpp_solver = _ConsensusADMM(self.num_agents, variable_dim, self.params)
            
            # Set up loss functions and gradients
            for i in range(self.num_agents):
                if self.local_losses[i] is not None and self.local_gradients[i] is not None:
                    self._cpp_solver.set_agent_loss(i, self.local_losses[i], self.local_gradients[i])
                
                if self.local_prox_ops[i] is not None:
                    self._cpp_solver.set_agent_proximal(i, self.local_prox_ops[i])
    
    def set_agent_loss(self, agent_id: int, loss_func: Callable, grad_func: Callable):
        """Set loss function and gradient for a specific agent."""
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}")
        
        self.local_losses[agent_id] = loss_func
        self.local_gradients[agent_id] = grad_func
        
        if self._cpp_solver is not None:
            self._cpp_solver.set_agent_loss(agent_id, loss_func, grad_func)
    
    def set_agent_proximal(self, agent_id: int, prox_func: Callable):
        """Set proximal operator for a specific agent."""
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"Invalid agent_id: {agent_id}")
        
        self.local_prox_ops[agent_id] = prox_func
        
        if self._cpp_solver is not None:
            self._cpp_solver.set_agent_proximal(agent_id, prox_func)
    
    def solve(self, 
              initial_x: Optional[List[np.ndarray]] = None,
              initial_z: Optional[np.ndarray] = None) -> Dict:
        """
        Solve the consensus ADMM problem.
        
        Parameters:
        -----------
        initial_x : list of np.ndarray, optional
            Initial local variables for each agent
        initial_z : np.ndarray, optional
            Initial consensus variable
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'consensus': consensus variable z
            - 'locals': list of local variables x_i  
            - 'duals': list of dual variables u_i
            - 'residuals': convergence history
            - 'converged': whether algorithm converged
            - 'iterations': number of iterations
        """
        # Determine variable dimension from initial values
        if initial_x is not None:
            variable_dim = initial_x[0].shape[0]
        elif initial_z is not None:
            variable_dim = initial_z.shape[0]
        else:
            raise ValueError("Must provide either initial_x or initial_z to determine variable dimension")
        
        # Initialize C++ solver
        self._initialize_cpp_solver(variable_dim)
        
        # Set initial values if provided
        if initial_x is not None and initial_z is not None:
            self._cpp_solver.initialize(initial_x, initial_z)
        elif initial_x is not None:
            # Initialize z as average of x_i
            initial_z = np.mean(initial_x, axis=0)
            self._cpp_solver.initialize(initial_x, initial_z)
        elif initial_z is not None:
            # Initialize x_i as z
            initial_x = [initial_z.copy() for _ in range(self.num_agents)]
            self._cpp_solver.initialize(initial_x, initial_z)
        
        # Solve
        results = self._cpp_solver.solve()
        
        # Convert to Python dictionary format
        return {
            'consensus': results.consensus,
            'locals': results.local_variables,
            'duals': results.dual_variables, 
            'primal_residuals': results.primal_residuals,
            'dual_residuals': results.dual_residuals,
            'converged': results.converged,
            'iterations': results.iterations
        }
    
    def update_params(self, **kwargs):
        """Update ADMM parameters."""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        if self._cpp_solver is not None:
            self._cpp_solver.set_params(self.params)