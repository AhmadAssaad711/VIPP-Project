* on the topic of RL controllers
	* learning to tune pure pursuit in autonomous racing: Joint Lookahead and Steering-Gain Control with PPO, mohamad elgouhary and Amr S. El-wakeel


* first question I want to ask is how does my simulations work on varying environments, so i train on an env then test on another...
	* so i gotta create a new randomized environment to test my agent on, i continue training on that same static environment tho
	* create class RandomTrackGenerator
		* then use that class instead of the normal one


* comparing the basic and physics informed RL agents, we notice that they perform extremely similar to generalized environments, what could that show? the physics informed agent memorized the track and was unable to generalize while the standard agent was able to generalize
	* why?
	* how can you explain how changing the state, action, and reward functions affect learning and the process of...

back to the paper:
* their goal is to use RL to learn the lookahead distance and velocity gain
* what I want to try is using a reisdual controller and to test how well it performs
* 
