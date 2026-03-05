- environment work:
	- testing out different reward functions (ablation study)
	- also i tested the agent's ability to generalize
	- where the more basic agent generalized much better than the PIRL
		- might be due to the hyperparameters and constants we talked about, where the agent memorizes the track
		- so tuning that, training takes them
	- also, part of my learning includes formally understanding interpretable RL policies

now literature review:
- read the paper you sent me
	- they applied PPO to estimate lookahead distance and steering gain, that was the main contribution separating lookahead distance and steering gain as two separate parameters
	- but again, they used a constant velocity, which i think reinforces the idea of a longitudinal controller

* more literature review:
	* no focus on longitudinal controllers
	* only paper I found used a PID controller and tried to learn the parameters
	* however, professor I want to ask whether there is a motivation to introduce RL as a longitudinal controller, this is something i have yet to answer and plan to answer throughout next week 

- finally, related to the project of yasmine 
	- using rl for hughlevel waypoint generation... kind of lostpoiu\