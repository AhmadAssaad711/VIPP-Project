### playing with the environment and reward function

* addressing the problem that agent memorized rather than generalized
	* physics priors in 'curvature' when tasked against new test envrionemtns did not know how to generalize and basic agent generalized better


		* so want to change a couple of things
		* curvature is currently map-centric defined, then agent would memorize rather than generalize


#### changing observation space

* including curvature change
* question is, how can curvature help training?

- different Q-table sizes, unfair....

### changing reward function

* an ablation study, how does each term affect the reward function






* 



==key ideas to look out for:==

* convergence of the Q-table, how much of the Q-table has been visited
* also want to do an ablation study for the parameters weights and terms
* 