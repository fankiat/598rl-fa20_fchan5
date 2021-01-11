# HW1

## What to do

Your goal is to implement five reinforcement learning algorithms. All of these algorithms are "tabular" and can be applied only to systems with finite state and action spaces.

Two algorithms are *model-based* and require knowledge of the state transition function:
* Value iteration (Chapter 4.4, Sutton and Barto)
* Policy iteration (Chapter 4.3, Sutton and Barto)

Two algorithms are *model-free* and require only the ability to simulate the state transition function:
* SARSA with an epsilon-greedy policy, i.e., on-policy TD(0) to estimate Q (Chapter 6.4, Sutton and Barto)
* Q-learning with an epsilon-greedy policy, i.e., off-policy TD(0) to estimate Q (Chapter 6.5, Sutton and Barto)

One final algorithm, which is also model-free, computes the value function that is associated with a given policy (and does not by itself learn an optimal policy):
* TD(0) to estimate V (Chapter 6.1, Sutton and Barto)

We have provided two environments with which to test your algorithms:
* A grid-world (Example 3.5, Chapter 3.5, Sutton and Barto) - see [simulation code](gridworld.py) and [example of how to use simulation code](test_gridworld.py)
* A simple pendulum with discretized state and action spaces (e.g., http://underactuated.mit.edu/pend.html) - see [simulation code](discrete_pendulum.py) and [example of how to use simulation code](test_discrete_pendulum.py)

For both environments, you should express the reinforcement learning problem as a Markov Decision Process with an infinite time horizon and a discount factor of $\gamma = 0.95$.

Please apply value iteration, policy iteration, SARSA, and Q-learning to the grid-world, for which an explicit model is available. Please apply TD(0) to learn the value function associated with the optimal policy produced by SARSA and by Q-learning.

Please apply only SARSA and Q-learning to the simple pendulum with discretized state and action spaces, for which an explicit model is not available. Again, please apply TD(0) to learn the value function associated with the optimal policy produced by these other two algorithms.

## What to submit (due by 10am on Tuesday, September 15)

Create a branch of our `598rl_fa20` github repository. Call this branch `hw1_yournetid`, with "`yournetid`" replaced by your own NetID in **lower-case letters**. In this branch, create the directory `./homework/hw1/hw1_yournetid` in our github repository. This directory should contain all of your code (including a copy of both `gridworld.py` and `discrete_pendulum.py`). It should also contain a file called `README.md` with your results and with instructions for how to run your code to recreate your results.

At minimum, the `README.md` should contain the following results for both the grid-world and the simple pendulum with discretized state and action spaces:
* One plot that contains learning curves:
    - For both value iteration and policy iteration (grid-world only), plot the mean of the value function $\frac{1}{25}\sum_{s = 1}^{25} V(s)$ versus the number of *value* iterations. For value iteration, "the number of value iterations" is synonymous with the number of iterations of the algorithm. For policy iteration, where the algorithm alternates between policy evaluation and policy improvement, "the number of value iterations" means the number of iterations spent on policy evaluation (which dominates the computation time).
    - For both SARSA and Q-learning (both environments), plot the total discounted reward versus the number of simulation steps.
* For each trained agent, one plot with an example trajectory
* One plot that contains learning curves for SARSA, for several different values of epsilon
* One plot that contains learning curves for SARSA, for several different values of alpha
* One plot that contains learning curves for Q-learning, for several different values of epsilon
* One plot that contains learning curves for Q-learning, for several different values of alpha
* A visualization of the policy that corresponds to each trained agent
* A visualization of the value function - learned by TD(0) - that corresponds to each agent trained by SARSA and Q-learning

Please also discuss (briefly) the relative performance of each algorithm.

Remember that we expect each of you to write your own code from scratch.

Final submission should be by pull request.
