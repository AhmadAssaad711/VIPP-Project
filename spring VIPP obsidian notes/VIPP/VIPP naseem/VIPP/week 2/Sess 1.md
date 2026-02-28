Plan:
* ==deep learning (spinning up AI gym)==
* continue background research on the effect of combining an RL controller with a geometric one
	* spinning up, policy gradient....????


* policy based methods look at tuning a policy J parameterized by thetas which represent the expected cumulative discounted reward
	* the big question is how do i tune these parameters
	* gradient descent
	* but the bigger question is in computing the gradient algorithmically

from that we really need to ask, what are we even optimizing

* so first we start from the probability of a trajectory which is the multiplicative sum of transition probability and policy parameterized by thetas
* then the **gradient** of probability of trajectory parameterized by theta depends on that probability itself and the log
* then substituting, we find that the rate of change of return (reward is easy, probability of a trajectory is difficult)
* where we eventually reach from a trajectory to a sum over different timesteps, that the rate of change of the return following a policy pi is 




key math intuition:
* contains the gradient of the probability of trajectory, strip away the dynamics, keep it policy dependent (using log-derivative rule)
* vanilla policy gradient relies on the computed returns, a value function could be added, will be called advantage, is not bootstrapping, the value function will need to become more accurate and updated

---

Trust Region Policy Optimization (TRPO) and (PPO):
* both work in a way to minimize major changes in policy
* TRPO solves second order constraint equations while PPO just uses computation by bounding the maximum the policy is allowed to change by
* now both have slight differences in their math
* but i think the big idea is enough and implementation will come on its own