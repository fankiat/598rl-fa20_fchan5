# HW4

## What to do

Your goal is to implement the PPO algorithm. There are two key differences between PPO and REINFORCE (which you implemented in [HW3](../hw3)), both of which are policy gradient methods:

* PPO weights gradient terms with advantage estimates instead of with total reward (or reward-to-go). These advantage estimates, in turn, require value function estimates, which are obtained by function approximation with a so-called *critic*. Therefore, we call PPO an *actor-critic method*, as a special case of policy gradient method.

* PPO uses a slightly modified loss function that discourages taking policy gradient steps that are too large. We have seen that, despite the use of importance sampling, it is in general a bad idea to let the current policy diverge too much from the one that was used to sample the most recent batch of trajectories.

Like REINFORCE, PPO can be applied to systems with continuous state and action spaces. Unlike REINFORCE, PPO can be expected to actually work when applied to these systems (given a reasonable number of simulation steps).

PPO is described in the following paper:

> J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017), "Proximal Policy Optimization Algorithms," URL: https://arxiv.org/abs/1707.06347

Note that the paper has never appeared in an archival journal, yet has been cited many thousands of times and describes a method that is used throughout the RL literature.

In this paper, you will find a derivation of PPO, pseudocode, and details of implementation. You will also find a list of hyperparameter values that were used by the authors. You may find that certain details are either missing or need to be changed, so should expect that it may take you some time to figure out exactly what to implement.

Please test your algorithm with the simple pendulum, this time with fully-continuous state and action spaces. See the [simulation code](pendulum.py) and [example of how to use the simulation code](test_pendulum.py). By default, this pendulum environment uses a "dense reward" instead of the sparse reward we have assumed in prior versions of the pendulum. We suggest you apply PPO first to the pendulum with a dense reward, and then - if that is successful - to the pendulum with a sparse reward.


## What to submit (due by 10am on Tuesday, December 1)

Create a branch of our `598rl_fa20` github repository. Call this branch `hw4_yournetid`, with "`yournetid`" replaced by your own NetID in **lower-case letters**. In this branch, create the directory `./hw4/hw4_yournetid` in our github repository. This directory should contain all of your code (including a copy of `pendulum.py`). It should also contain a file called `README.md` with your results and with instructions for how to run your code to recreate your results.

At minimum, the `README.md` should contain the following results for PPO applied to the "dense reward" version of pendulum:

* One plot that contains a learning curve for the actor (the total reward versus the number of simulation steps)
* One plot that contains a learning curve for the critic (the MSE loss versus the number of simulation steps)
* One plot with an example trajectory
* One animated gif with an example trajectory
* A visualization of the policy
* A visualization of the value function

In the best submissions, the two learning curves will be averaged over many runs of PPO, and similar results will also be shown for the "sparse reward" version of pendulum.

As usual, the `README.md` should contain a brief discussion of your results.

Remember that we expect each of you to write your own code from scratch.

Final submission should be by pull request.

## Code review (due by 10am on Tuesday, December 8)

You are responsible for reviewing the code of at least one colleague. In particular, you should:
* Choose a [pull request](https://github.com/compdyn/598rl-fa20/pulls) that does not already have a reviewer, and assign yourself as a reviewer. **Do this no later than Tuesday, December 1.**
* Perform a review. **Do this no later than Friday, December 4.**
* Improve your own code based on reviews that you receive. Respond to every comment. If you address a comment fully (e.g., by changing your code), you mark it as resolved. If you disagree with or remain uncertain about a comment, engage in follow-up discussion with the reviewer on github. (Reply to this follow-up on code you reviewed as well!) **Do this no later than 10am on Tuesday, December 8.**

The goal of this review process is to arrive at a version of your code that is functional, reasonably efficient, and easy for others to understand. The goal is *not* to make all of our code the same (there are many different ways of doing things). The goal is also *not* to grade the work of your colleagues - your reviews will have no impact on others' grades.

In your reviews, don't forget to remind your colleagues to do the simple things like name their PR correctly ("Submit hw4 for Firstname Lastname (netid)") and include their name in their README!

Here are some resources that may be helpful:
* [Github docs on PR code reviews](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-reviews)
* [Google best practices for code review](https://google.github.io/eng-practices/review/)
* From Microsoft, a [blog post on code review](https://devblogs.microsoft.com/appcenter/how-the-visual-studio-mobile-center-team-does-code-review/) and a [study of the review process](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/MS-Code-Review-Tech-Report-MSR-TR-2016-27.pdf)
* From RedHat, a [python-specific blog post on code review](https://access.redhat.com/blogs/766093/posts/2802001)
* [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf), a modern classic

Please let us know if you have a favorite resource that you think should be added to this list.
