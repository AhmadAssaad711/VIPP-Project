### playing with the environment and reward function

* addressing the problem that agent memorized rather than generalized
	* physics priors in 'curvature' when tasked against new test envrionemtns did not know how to generalize and basic agent generalized better


		* so want to change a couple of things
		* curvature is currently map-centric defined, then agent would memorize rather than generalize

### memorization vs generalization




#### changing observation space

* including curvature change
* question is, how can curvature help training?

- different Q-table sizes, unfair....

### changing reward function

* an ablation study, how does each term affect the reward function
	* normal reward function: 3 terms norm error, alive rewad, and jerk

![[error_per_step_vs_training.png]]


![[path_following_q_learning.png]]





![[steps_vs_training.png]]

---
#### no alive reward:


![[error_per_step_vs_training 1.png]]

![[policy_heatmap.png]]


### jerk cost


- performance did not shift greatly, but it did shift nonetheless
- less steps per episode



## PIRL

![[Pasted image 20260304153042.png]]

- still even after changing the training data, bad performance
- but i think there is a fundemental error, that is with the constant lookahead distance, really is a big reason why the first formulation worked so much more better


==key ideas to look out for:==

* convergence of the Q-table, how much of the Q-table has been visited
* also want to do an ablation study for the parameters weights and terms
* 