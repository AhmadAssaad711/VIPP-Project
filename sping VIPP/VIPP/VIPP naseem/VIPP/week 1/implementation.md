* evaluation metrics will always be:
	* average error per step
	* steps per episode
	* average return

the faster it learns, faster it converges

* goal is to minimize the path following error


**TODO**:
* finish physics informed agent
* outline of presentation + updating github repo
* writing a report on this (weekend)



* learning is slow, how do i boost learning???, lets look at the reward function
* lets also check the learned policy



* in the normal agent, it is cutting corners
* in PIRL aim is to make learning faster, no cutting corners, new action space
* i think the only thing to play with now remains the reward function
	* trying out without the heading angle, causing interference

* problems with current code:
	* state return, kappa_vehicle not kappa_err
	* reward function accounts for error twice





