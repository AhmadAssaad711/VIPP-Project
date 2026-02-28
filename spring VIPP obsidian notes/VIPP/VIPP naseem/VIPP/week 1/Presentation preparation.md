* Introduction:
	* my research interests is currently using learning methods to solve control problems
	* motivation is that many unsolved problems in control from fully autonomous robots to cars require policies that **adapt** to their environments, this is where learning comes in

What we will discuss today:
**A brief Overview of Reinforcement learning**
* 1 slide, mdp terminology state action reward policy return
	 * value function
	 * state action
* 1 slide breaking down model based, model free
	* model based, where my agent has access to the transition probability, so here we can imagine that the agent can **predict** instead of needing to try out different experiences it can predict where a certain state and action can lead it
	* dynamic programming: bellman equation, can literally be solved using recursion

	* model free: all have the common theme of needing to use past trajectories
	* 
		* a common theme is that it involves continuosly updating my estimation of how good or bad a state is depedning on experience
		* the flow of how it was developed was from MC to temporal difference
		* key difference is instead of updating my values after every episode i do it after every action
		* further developments are Q-learning
			* then we look at deep learning methods, not tabular it becomes function approximation
			* used for continuous problems

	* the bellman equation
		* we first write out the return in recursive form 
		* then we relate the value at a next state to the value at this state
		* notice this idea of estimation...

	* Q-learning: what we will be using
		* is a giant look up table
			* for every state action combo, there is a probability or 'index' of how good or bad it is
		* discrete observation and action spaces
		* the Q(s,a) update
		* this is a tabular based, instead of learning the actual policy it just updates and learns it

* Onto the second step:
	* we want to briefly introduce the kinematic model
	* we assume car is squished into 2 tires and an axle
	* geometry becomes easier, equations of motion are...
		* main intuition is that there is something called an instantaneous center of rotation where we can define the velocity of anything as angular velocity times radius
		* then we find the yaw rate

Now we are ready to tackle the issue of path following using basic RL

* intuitively, we want to compare 2 different agents and the policies they learn:
* **basic agent**
	* we want to follow a path, im too much to the left go right too much right go left
	* there is a sense of how much i need to go right or left
	* so how can we represent this mathematically:
		* we need observation space
		* action space
		* reward function
	* lateral error and heading error
	* steering angle
	* energy function that discourages error

**physics informed**
* Now a brief introduction to what we mean by physics informed and what classifies as physics informed
* this is a middle ground between model free and model based, we inject the information we have to help it learn
* can be in the observation, action, and reward
* best way i can explain this is through the work

* now the main themes in which physics in injected is through taking away illogical actions such as turning 120 degrees (respecting actuator limits)
* another way may be by shaping the problem in a way that adds more information to the state that would help it learn (in addition to it being markovian)
	* such as adding curvature
	* from our policy analysis we notice hard turning, curvature should ease that
	* reward function now also incorporates error in curvature, that is we want the vehicle to maintain the same curvature as the road

Analysis of the policy:
* now we need to note that learning took much longer, extra observation spaces
* still needs tuning
* did it learn...????

Next steps:
* i need to continue tuning the physics informed agent to make a detailed report  on a comparision between a basic RL agent and a physics informed agent 
* new topic may be to use RL as a residual controller, that is to minimize the error from another physics based model such as pure pursuit
* a more realistic environment, friction forces and noise
* finally as a caveat, one of the topics im interested in learning about throughout this journey is not only what did RL learn but why..... down the road of explainable AI

