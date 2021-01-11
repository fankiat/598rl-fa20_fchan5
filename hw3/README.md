# HW3

## What to do

Your goal is to implement the REINFORCE algorithm. The key difference between REINFORCE and DQN (which you implemented in [HW2](../hw2)) is that REINFORCE learns the *policy* rather than the action-value function Q. It can also be applied to systems with a continuous state space and (unlike DQN) a continuous action space, although you will focus in this assignment on its application to a system with finite state and action spaces.

#### 1) Apply REINFORCE to the GridWorld environment

Consider the same [gridworld environment](gridworld.py) that you considered in [HW1](../hw1). Implement the REINFORCE algorithm with a tabular policy, using softmax to compute action probabilities, using auto-differentiation to compute payoff gradients (the [categorical distribution](https://pytorch.org/docs/stable/distributions.html#categorical) may be helpful, as we discussed in class), and using the [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) without momentum to take descent steps. Assume that reward is **not** discounted.

#### 2) Importance sampling

REINFORCE is an on-policy algorithm. Importance sampling can be used to produce an off-policy variant of this algorithm. The benefits of using off-policy learning in policy gradient methods are similar to those you discovered in the context of value function methods like DQN with "experience replay." Keep a history of state-action pairs (along with weights) and choose a subset of them for mini-batch training at each iteration of REINFORCE. We will discuss the details of this extension to REINFORCE in class on Thursday, October 15.

#### 3) Causality

REINFORCE depends on having a good estimate of the policy gradient. This estimate is obtained by computing a sample mean - the average gradient over some number of sampled trajectories. If the variance of the gradient is high, then many samples are required to obtain a good estimate of the mean. One way to reduce the variance is to eliminate any term in the gradient whose mean is known to be identically zero. (If you know a quantity is zero in expectation, why estimate it?) The idea of "causality" is that the reward at a given time should not - in expectation - depend on actions taken at any later time. Modify your computation of the gradient to eliminate terms that are zero due to "causality." We will discuss the details of this extension to REINFORCE in class on Thursday, October 15.

#### 4) Baseline shift

It is possible to show that the gradient of `J(theta)` and the gradient of `J(theta) - b` have the same mean for any b but can have different variance (when sampled). "Baseline shift" means choosing b to minimize the variance. Modify your computation of the gradient to incorporate baseline shift. Remember that b is chosen element-wise - a different value of b is often best for each component of theta. We will discuss the details of this extension to REINFORCE in class on Thursday, October 15.

#### 5) Investigation of your choice

Do at least **one** of the following things:

* Compare the sample variance of the policy gradient with and without any extensions to REINFORCE. In particular, verify that "causality" and "baseline shift" reduce the sample variance while keeping the sample mean the same. (Does "importance sampling" affect the sample variance?)
* Test your algorithm on both the "easy" version of gridworld (deterministic state transition) and the "hard" version (stochastic state transition).
* Compare your results when gradient steps are implemented with the [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) and with the [Adam optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
* Apply REINFORCE with a tabular policy to the [discrete pendulum environment](discrete_pendulum.py), with finite state and action spaces, that you considered in [HW1](../hw1).
* Apply REINFORCE to the [pendulum environment](pendulum.py), with continuous state and action spaces, assuming the use of a Gaussian policy that is described by a neural network. By default, this pendulum environment uses a "dense reward" instead of the sparse reward we have assumed in prior versions of the pendulum. Beware! Even with a bug-free implementation and a good choice of hyper-parameters, it is likely that you will need a very large number of simulation steps in order for REINFORCE to produce good results (e.g., tens of millions). We will be impressed if you get this to work (and astonished if you get it to work with sparse reward).

We may discuss some of these extensions in class next week (October 20 and 22).


## What to submit (due by 10am on Tuesday, October 27)

Create a branch of our `598rl_fa20` github repository. Call this branch `hw3_yournetid`, with "`yournetid`" replaced by your own NetID in **lower-case letters**. In this branch, create the directory `./hw3/hw3_yournetid` in our github repository. This directory should contain all of your code (including a copy of `gridworld.py` and, optionally, of `pendulum.py`). It should also contain a file called `README.md` with your results and with instructions for how to run your code to recreate your results.

At minimum, the `README.md` should contain the following results for REINFORCE (with no extensions - no importance sampling, no causality, and no baseline shift) applied to the "easy version" of gridworld:

* A plot that contains a learning curve (the total reward versus the number of simulation steps)
* A visualization of the policy (compared to the optimal policy, if known)

At minimum, the `README.md` should also contain the results of an ablation study that compares the performance of REINFORCE (as quantified by learning curves) under the following conditions:

* With importance sampling, causality, and baseline shift
* With causality and baseline shift
* With importance sampling and baseline shift
* With importance sampling and causality

Your results are likely to vary from one training run to the next. You will be able to draw stronger conclusions from your ablation study if you average your results over several training runs. Please discuss (briefly) both the design of your ablation study and your conclusions.

Finally, the `README.md` should contain the results of your chosen investigation(s).

Remember that we expect each of you to write your own code from scratch.

Final submission should be by pull request.

## Code review (due by 10am on Thursday, November 5)

You are responsible for reviewing the code of at least one colleague. In particular, you should:
* Choose a [pull request](https://github.com/compdyn/598rl-fa20/pulls) that does not already have a reviewer, and assign yourself as a reviewer. **Do this no later than Tuesday, October 27.**
* Perform a review. **Do this no later than Friday, October 29.**
* Improve your own code based on reviews that you receive. Respond to every comment. If you address a comment fully (e.g., by changing your code), you mark it as resolved. If you disagree with or remain uncertain about a comment, engage in follow-up discussion with the reviewer on github. (Reply to this follow-up on code you reviewed as well!) **Do this no later than 10am on Thursday, November 5.**

The goal of this review process is to arrive at a version of your code that is functional, reasonably efficient, and easy for others to understand. The goal is *not* to make all of our code the same (there are many different ways of doing things). The goal is also *not* to grade the work of your colleagues - your reviews will have no impact on others' grades.

In your reviews, don't forget to remind your colleagues to do the simple things like name their PR correctly ("Submit hw3 for Firstname Lastname (netid)") and include their name in their README!

Here are some resources that may be helpful:
* [Github docs on PR code reviews](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-reviews)
* [Google best practices for code review](https://google.github.io/eng-practices/review/)
* From Microsoft, a [blog post on code review](https://devblogs.microsoft.com/appcenter/how-the-visual-studio-mobile-center-team-does-code-review/) and a [study of the review process](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/MS-Code-Review-Tech-Report-MSR-TR-2016-27.pdf)
* From RedHat, a [python-specific blog post on code review](https://access.redhat.com/blogs/766093/posts/2802001)
* [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf), a modern classic

Please let us know if you have a favorite resource that you think should be added to this list.
