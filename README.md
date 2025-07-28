# gradient-free-optimization
Reinforcement Learning Project: Comparing Gradient-Free Optimization Methods This script implements and compares two gradient-free optimization methods: 1. Zeroth-order Optimization 2. Population Methods


Reinforcement Learning Project Report: Gradient-Free Optimization Methods
1. Introduction
This project implements and compares two gradient-free optimization methods for training a reinforcement learning agent in the LunarLanderContinuous environment. The goal is to maximize the expected return by optimizing a neural network policy without using gradient-based methods.
2. Methods Implemented
2.1 Policy Representation
•
Neural Network Architecture:
o
Input layer: 8 neurons (matching state space dimensions)
o
Hidden layer: 128 neurons with tanh activation
o
Output layer: 2 neurons with tanh activation (matching action space dimensions)
•
Uses PyTorch's automatic differentiation capabilities
2.2 Gradient-Free Optimization Techniques
Zeroth-Order Optimization
1. Algorithm:
o
Generates two parameter perturbations (θ+ and θ-) using Gaussian noise
o
Evaluates both perturbations in the environment
o
Computes a gradient estimate: `0.5 * (R(θ+) - R(θ-)) * perturbation / noise_std`
o
Updates parameters using this pseudo-gradient
2. Key Characteristics:
o
Requires only function evaluations (no gradients)
o
More sample-efficient than population methods
o
Still sensitive to learning rate choice
Population-Based Methods
1. Algorithm:
o
Generates N random perturbations of current parameters
o
Evaluates each candidate policy in the environment
o
Selects the best-performing candidate
o
Updates policy parameters to the best candidate
2. Key Characteristics:
o
More exploration of parameter space
o
Less sensitive to local optima
o
Computationally expensive (requires many evaluations)
3. Implementation Details
3.1 Environment
•
LunarLanderContinuous-v3 from OpenAI Gym
•
State space: 8 continuous values
•
Action space: 2 continuous values (main engine, side engine)
3.2 Training Process
•
Each method runs for 25,000 episodes (50,000 total)
•
Evaluation uses 3 episodes per parameter set for stability
•
Results logged as "RETURN <episode> <value>" for analysis
3.3 Hyperparameters
•
Noise standard deviation: 0.1
•
Learning rate (Zeroth-order): 0.01
•
Population size: 50 candidates
•
Hidden layer size: 128 neurons
4. Key Reinforcement Learning Concepts
1. Policy Optimization:
o
Directly optimizes the policy function π(a|s)
o
Maximizes expected return rather than learning value functions
2. Exploration-Exploitation Tradeoff:
o
Zeroth-order: Exploits local gradient information
o
Population: Explores wider parameter space
3. Credit Assignment:
o
Both methods use complete episode returns
o
No discounting applied (could be added with γ parameter)
5. Results Analysis
The implementation produces:
1. Learning Curves:
o
Episode-by-episode return values
o
Smoothed for better visualization
o
Comparison plot of both methods
2. Performance Metrics:
o
Final achieved returns
o
Sample efficiency comparison
o
Training stability
6. Conclusion
   This project demonstrates two fundamental approaches to gradient-free policy optimization in reinforcement learning. The Zeroth-order method provides a more directed search using pseudo-gradients, while population methods offer broader exploration at higher computational cost. The implementation successfully applies these methods to the challenging LunarLanderContinuous environment, providing insights into their relative strengths and weaknesses.
