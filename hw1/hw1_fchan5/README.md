# HW1 (Fan Kiat Chan, fchan5)

## How to run the code
The algorithms for policy iteration, value iteration, sarsa and Q-learning with TD(0) for value function are implemented in `learning_algorithms.py`. These algorithms are called in [`gridworld_training.py`](gridworld_training.py) and [`pendulum_training.py`](pendulum_training.py), where the resulting values, policies and reward trajectories are saved after computation. These results are then later recovered for plotting the various analytics as below through [`gridworld_run.py`](gridworld_run.py) and [`pendulum_run.py`](pendulum_run.py). To recreate the data generated for each environment, one can first run `*_training.py` files to generate relevant policies, value functions amd reward trajectories, followed by `*_run.py` to generate the different plots.

## Grid-world
### 1. Learning curves for different algorithms
From the model-based method learning curve, we see that value iteration converges much faster than policy iteration. This seems reasonable since in value iteration algorithm, policy evaluation is truncated (stopped after just one sweep) instead of computed iteratively as in policy iteration.
![gridworld-learning-curves](./gridworldData/images/gridworld_learning_curve.png)

### 2. Example trajectories for different algorithms
The different method used provides policies that are able to find the optimal solution at the longer loop with reward portal at tile 1. However, ε can affect the agent to explore and sometime enter the shorter loop with reward portal at tile 3. This is illustrated in the policy visualization later.
![gridworld-trajectories](./gridworldData/images/gridworld_trajectory_different_method.png)

### 3. SARSA for different values of ε
Here we observe that smaller ε result in higher total discount reward in the long run. Too much exploration may result in the agent to pingpong between the two soluitions and as a consequence, total discount reward averaged over 10 episodes suffers.
![gridworld-sarsa-different-epsilon](./gridworldData/images/gridworld_sarsa_different_epsilon_alpha0.1.png)

### 4. SARSA for different values of α
![gridworld-sarsa-different-alpha](./gridworldData/images/gridworld_sarsa_different_alpha_epsilon0.1.png)

### 5. Q-learning for different values of ε
![gridworld-qlearning-different-epsilon](./gridworldData/images/gridworld_qlearning_different_epsilon_alpha0.1.png)

### 6. Q-learning for different values of α
![gridworld-qlearning-different-alpha](./gridworldData/images/gridworld_qlearning_different_alpha_epsilon0.1.png)

### 7. Visualization of policy and value function
In the plots below, the color contours shows the value function obtained for different methods (for SARSA and Q-learning, the value function is learned using TD(0)). The arrows show the action obtained from the policy (greedy). We note that while all the methods achieved policies that draws the agent towards tile 1 (bright yellow tile), we see that with Q-learning there is a short detour if the agent were to land itself on tile 2, whereby it will be drawn into the shorter loop at tile 3, but eventually escapes the loop right away and converges towards the longer loop.
![gridworld-visualize-policy-value](./gridworldData/images/gridworld_visualize_policy_value.png)

## Pendulum
The computational details for this study is done at (`n_theta`, `n_thetadot`, `n_tau`) = (31, 31, 31). The algorithms are allowed to run for 50,000 episodes, each episode with 100 simulation steps, resulting in a total of 5,000,000 simulation steps.

### 1. Learning curves for different algorithms
![pendulum-learning-curves](./pendulumData/images/pendulum_learning_curve.png)

### 2. Example trajectories for different algorithms
![pendulum-trajectories](./pendulumData/images/pendulum_trajectory_different_method.png)

### 3. SARSA for different values of ε
![pendulum-sarsa-different-epsilon](./pendulumData/images/pendulum_sarsa_different_epsilon_alpha0.1.png)

### 4. SARSA for different values of α
![pendulum-sarsa-different-alpha](./pendulumData/images/pendulum_sarsa_different_alpha_epsilon0.1.png)

### 5. Q-learning for different values of ε
![pendulum-qlearning-different-epsilon](./pendulumData/images/pendulum_qlearning_different_epsilon_alpha0.1.png)

### 6. Q-learning for different values of α
![pendulum-qlearning-different-alpha](./pendulumData/images/pendulum_qlearning_different_alpha_epsilon0.1.png)

### 7. Visualization of policy
In the plots below, the color contours shows the the torque (action) at different states, obtained using different model-free methods.
![pendulum-visualize-policy-value](./pendulumData/images/pendulum_visualize_policy.png)

### 8. Visualization of value function
Below are the value function learned by TD(0) corresponding to the agents with policy visualized above, trained by SARSA and Q-learning. We note the brighter spot in the middle of the contour (namely higher value function) corresponds to where we would expect the agent would aim to be, at small angles and angular velocity, which effectively translating physically to maintaining "upright".
![pendulum-visualize-policy-value](./pendulumData/images/pendulum_visualize_value.png)