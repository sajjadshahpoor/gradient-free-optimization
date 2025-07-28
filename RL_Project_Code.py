"""
Sajjad SHAHPOOR Project

Reinforcement Learning Project: Comparing Gradient-Free Optimization Methods
This script implements and compares two gradient-free optimization methods:
1. Zeroth-order Optimization
2. Population Methods

The implementation is for the LunarLanderContinuous environment from OpenAI Gym.
Results are saved to separate text files and plotted at the end.
"""

# Import necessary libraries
import numpy as np  # For numerical operations
import torch  # PyTorch for neural networks
import torch.nn as nn  # Neural network modules
import gymnasium as gym  # OpenAI Gym for RL environments
from typing import List  # For type hints
import random  # For random number generation
import os  # For file operations
import matplotlib.pyplot as plt  # For plotting results

# Set random seeds for reproducibility
SEED = 42  # Fixed seed value
torch.manual_seed(SEED)  # Set PyTorch random seed
np.random.seed(SEED)  # Set NumPy random seed
random.seed(SEED)  # Set Python random seed


#Neural network that decides how the lunar lander should act.
class PolicyNetwork(nn.Module):
    """Neural network policy with one hidden layer for LunarLanderContinuous."""
# 1. Architecture: Simple 2-layer neural network (input → hidden → output)
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        # Initialize the neural network
        super(PolicyNetwork, self).__init__()  # Initialize parent class
# 2. Input: 8-dimensional state vector (lander position, velocity, etc.)
        self.fc1 = nn.Linear(state_dim, hidden_size) # Input layer
        # Output layer from hidden units to actions
# 3. Output: 2-dimensional action vector (engine throttles)
        self.fc2 = nn.Linear(hidden_size, action_dim) # Output layer
        # Tanh activation for bounded action outputs (-1 to 1)
# 4. Tanh Activation: Ensures actions stay between -1 and 1 (required for continuous control)
        self.activation = nn.Tanh() # Activation function
# 5. Parameters: ~10,000 trainable weights (128 hidden units)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network
        x = self.activation(self.fc1(x))  # Hidden layer with tanh
        x = torch.tanh(self.fc2(x))       # Output layer with tanh for bounded actions (Bounded actions (-1 to 1))
        return x

# Tests how well the current policy performs in the environment.
def evaluate_policy(env: gym.Env, policy: PolicyNetwork, n_episodes: int = 1, render: bool = False) -> float:
    # Evaluates a policy by running it in the environment
    total_return = 0.0  # Accumulator for total return
# 1. Resets environment to start new episode
    for _ in range(n_episodes):  # Run for specified number of episodes

        # - ** (Markov Decision Process) MDP - > States, Actions, Rewards, Transitions (P(s’|s,a)), Discount factor
        state, _ = env.reset()  # Reset environment at start of episode   -**  "MDP (State) Initial state (8D vector)"
        episode_return = 0.0  # Accumulator for episode return
        done = False  # Flag for episode completion
# 2. Runs policy through full episode (until termination)
        while not done:  # Run until episode ends
            if render:  # Optionally render the environment
                env.render()
                
            # Convert state to tensor and get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert to PyTorch tensor
            with torch.no_grad():  # Disable gradient calculation for evaluation
                action = policy(state_tensor).numpy()[0]  # Get action from policy  -  ** " MDP (Action) Decision based on current state"
# 3. Tracks cumulative reward (return)
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action) #  -  **  "MDP (State) Next state after action"
## The 'reward' variable contains the punishment/prize
            done = terminated or truncated  # Episode ends if terminated or truncated
            episode_return += reward  # Accumulate reward - **  <-- Accumulates all punishments/rewards
            state = next_state  # Update current state
# 4. Averages results over multiple episodes for reliability
        total_return += episode_return  # Add episode return to total
# 5. Optional rendering for visualization
    return total_return / n_episodes  # Return average return across episodes


# Improves policy without using gradients by testing small perturbations.
def zeroth_order_optimization(env: gym.Env,
                             policy: PolicyNetwork,
                             noise_std: float = 0.05,    # Standard deviation of parameter perturbations
                             learning_rate: float = 0.02, # Step size for parameter updates
                             n_eval_episodes: int = 5,   # Number of episodes for evaluation
                             max_episodes: int = 100000, # Maximum total episodes
                             log_file: str = "zeroth_order_returns.log"): # File to save results

        """
        Observation after changing the parameters (Parameter Effect ):

        noise_std	0.05	↑ help escape local optima but risk overshooting.	↓ More precise but slower convergence
        learning_rate	0.02	↑ faster learning (but might overshoot good solutions)	↓ Slower but more stable training
        n_eval_episodes	5	↑ more reliable performance estimates (but slower)	↓ Faster but not reliable performance estimates
        """

    # Zeroth-order optimization (gradient estimation via finite differences)
    # Initialization
    episode_count = 0  # Counter for total episodes
    best_return = -float('inf')  # Track best return seen
    
    with open(log_file, 'w') as f:  # Open log file for writing
        while episode_count < max_episodes:  # Main training loop
            # 1. Evaluate current policy
            current_return = evaluate_policy(env, policy, n_eval_episodes)
            episode_count += n_eval_episodes  # Increment episode count
            print(f"Episode {episode_count}: Current Return = {current_return:.2f}")
            print(f"RETURN {episode_count} {current_return}", file=f)  # Log result
            
            # Track best return
            if current_return > best_return:
                best_return = current_return
            
            if episode_count >= max_episodes:  # Check termination condition
                break
            
            # Generate perturbation direction
            perturbation = []  # Store perturbations for each parameter
            original_params = []  # Store original parameters
            for param in policy.parameters():
                # Create random perturbation with specified standard deviation
                perturbation.append(torch.randn_like(param) * noise_std)
                original_params.append(param.clone())  # Save original parameters
# 1. Creates two opposite tweaks to policy weights
            # Create theta+ and theta- perturbations
            with torch.no_grad():  # Disable gradient tracking
                # theta+ = current + perturbation
                for param, pert in zip(policy.parameters(), perturbation):
                    param.add_(pert)  # Add perturbation to parameters
# 2. Tests both tweaked (Theta +/-) versions in environment
                # Evaluate theta+
                theta_plus_return = evaluate_policy(env, policy, n_eval_episodes)
                episode_count += n_eval_episodes
                print(f"RETURN {episode_count} {theta_plus_return}", file=f)
                
                # theta- = current - perturbation
                for param, pert in zip(policy.parameters(), perturbation):
                    param.sub_(2 * pert)  # Subtract 2*pert to get theta-
                
                # Evaluate theta-
                theta_minus_return = evaluate_policy(env, policy, n_eval_episodes)
                episode_count += n_eval_episodes
                print(f"RETURN {episode_count} {theta_minus_return}", file=f)
                
                # Reset to original parameters
                for param, orig in zip(policy.parameters(), original_params):
                    param.copy_(orig)  # Restore original parameters
# 3. Uses performance difference to estimate improvement direction
                # Compute gradient estimate using the difference in returns
                gradient_estimate = []
                for pert in perturbation:
                    # Finite difference gradient estimate
                    grad = 0.5 * (theta_plus_return - theta_minus_return) * pert / noise_std
                    gradient_estimate.append(grad)
# 4. Adjusts weights toward better-performing version
                # Update parameters using the gradient estimate
                for param, grad in zip(policy.parameters(), gradient_estimate):
                    param.add_(learning_rate * grad)  # Gradient ascent step
            
            print(f"Episode {episode_count}: Updated policy (theta+ {theta_plus_return:.2f}, theta- {theta_minus_return:.2f})")

# (Evolutionary Approach): Uses survival-of-the-fittest to evolve better policies.
def population_based_training(env: gym.Env, 
                            policy: PolicyNetwork,
                            population_size: int = 50,  # Number of candidate policies
                            noise_std: float = 0.1,    # Standard deviation of perturbations
                            n_eval_episodes: int = 3,  # Episodes per evaluation
                            max_episodes: int = 100000, # Maximum total episodes
                            log_file: str = "population_returns.log"): # Log file

        """
        Observation after changing the parameters (Parameter Effect ):

        population_size	50	↑ More exploration, slower training	↓ Faster but may miss good solutions
        noise_std	0.1	↑ Larger policy changes, more randomness	↓ Finer but slower improvements
        n_eval_episodes	3	↑ Smoother performance estimates	↓ Noisier but faster evaluations

        population methods outperform zeroth-order because Population methods benefit from more candidates (population_size).
      """

    
    # Population-based training (evolutionary strategy)
    best_return = -float('inf')  # Track best return
    episode_count = 0  # Episode counter
    
    with open(log_file, 'w') as f:  # Open log file
        while episode_count < max_episodes:  # Main training loop
            # Save original parameters
            original_params = [param.clone() for param in policy.parameters()]
            
            # 1. Evaluate current policy
            current_return = evaluate_policy(env, policy, n_eval_episodes)
            episode_count += n_eval_episodes
            print(f"Episode {episode_count}: Current Return = {current_return:.2f}")
            print(f"RETURN {episode_count} {current_return}", file=f)
            
            # Track best return
            if current_return > best_return:
                best_return = current_return
            # 2. Create population of variants (perturbed policies)
            population = []  # Store candidate policies
# Mutation: Creates 50 slightly randomized policy variants
            for _ in range(population_size):
                # Create a new policy with perturbed parameters
                perturbed_policy = PolicyNetwork(env.observation_space.shape[0], 
                                               env.action_space.shape[0])
                perturbed_policy.load_state_dict(policy.state_dict())  # Copy current policy
                
                # Add Gaussian noise to parameters
                with torch.no_grad():
                    for param in perturbed_policy.parameters():
                        param.add_(torch.randn_like(param) * noise_std)  # Add noise
                
                population.append(perturbed_policy)  # Add to population
# Selection: Tests all variants and keeps the top performer
            # 3. Evaluate all candidates
            returns = []  # Store returns for each candidate
            for candidate in population:
                cand_return = evaluate_policy(env, candidate, n_eval_episodes)
                episode_count += n_eval_episodes
                returns.append(cand_return)
                print(f"RETURN {episode_count} {cand_return}", file=f)
                
                # Check if we've reached max episodes
                if episode_count >= max_episodes:
                    break
            
            if episode_count >= max_episodes:
                break
# Repetition: Iteratively improves through generations
            # 4. Select best performer
            best_idx = np.argmax(returns)  # Index of best candidate
            if returns[best_idx] > current_return:
                # Update policy if better candidate found
                policy.load_state_dict(population[best_idx].state_dict())
                print(f"Episode {episode_count}: Updated policy with return {returns[best_idx]:.2f}")

                
# Creates comparative performance graphs.
def plot_results(methods: List[str] = ["zeroth_order", "population"]):
    # Plots the learning curves for the optimization methods
    
    plt.figure(figsize=(12, 6))  # Create figure with specified size
    
    for method in methods:  # Plot each method's results
        log_file = f"{method}_returns.log"  # Log file name
        episodes, returns = [], []  # Store data points
        
        # Read data from log file
        with open(log_file) as f:
            for line in f:
                if line.startswith("RETURN"):
                    parts = line.split()
                    episodes.append(int(parts[1]))  # Parse episode number
                    returns.append(float(parts[2]))  # Parse return value
        
        # Plot with smoothing
        if len(episodes) > 0:
            # Simple moving average for smoothing
            window_size = max(1, len(returns) // 20)  # Dynamic window size
            smoothed_returns = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
            
            # Plot both raw and smoothed data
            plt.plot(episodes, returns, alpha=0.2, color=f"C{methods.index(method)}")  # Raw (transparent)
            plt.plot(episodes[window_size-1:], smoothed_returns, label=method.replace("_", " ").title(), 
                    color=f"C{methods.index(method)}", linewidth=2)  # Smoothed
    
    plt.xlabel("Episode")  # X-axis label
    plt.ylabel("Return")  # Y-axis label
    plt.title("Comparison of Gradient-Free Optimization Methods")  # Plot title
    plt.legend()  # Show legend
    plt.grid()  # Add grid lines
    plt.tight_layout()  # Adjust layout
    
    # Save and show plot
    plt.savefig("optimization_comparison.png")  # Save to file
    plt.show()  # Display plot

def run_all_methods(max_episodes: int = 100000):
    # Runs all optimization methods and plots results
    
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True)  # LunarLander environment
    state_dim = env.observation_space.shape[0]  # Get state dimension
    action_dim = env.action_space.shape[0]  # Get action dimension
    
    # Run each method with a fresh policy
    print("\n=== Running Zeroth-Order Optimization ===")
    zo_policy = PolicyNetwork(state_dim, action_dim)  # Create new policy
    zeroth_order_optimization(env, zo_policy, max_episodes=max_episodes//2)  # Run ZO
    
    print("\n=== Running Population-Based Training ===")
    pop_policy = PolicyNetwork(state_dim, action_dim)  # Create new policy
    population_based_training(env, pop_policy, max_episodes=max_episodes//2)  # Run PBT
    
    # Close environment
    env.close()
    
    # Plot results
    plot_results()  # Generate comparison plot

if __name__ == "__main__":
    # Entry point when script is run directly
    # Run all methods and plot comparison
    run_all_methods(max_episodes=100000)

    print("\nSaving results to:", os.path.abspath(""))  # Show where results are saved