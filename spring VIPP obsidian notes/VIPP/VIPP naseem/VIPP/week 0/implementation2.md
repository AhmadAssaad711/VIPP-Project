* initial implementation was wrong, overfiiting
* need to redo it
* understand the documentation first


so what i did today:
* reformulated the problem + checked the documentation
* how the RL agent behaves
	* takes in errors in lane position and heading errors (it was either this or x,y and x,y of lane, but the problem was giving the agent global coordinates made it memorize the track)
	* action is to find the needed heading angle
	* low level controller defines yaw rate through a proportional controller based on reference and actual heading error
	* reward function is made up of 3 terms, in the form of an energy function for error is position and energy function for jerk and term to reward longevity
	* used Q-learning, discretized them into 20 bins 
	* dynamics of the environment are through highway env, *not randomized map*
		* inject physics in the observation space (curvature of the road, what do i expect to see?? also lets try to analyze how its learning....)
		* the dynamics of the controller follow from the RL model, so from needed yaw find yaw rate PID, and the next states are given through euler integration in the environment 
	* next steps:
		* check out the randomness
		* add physics information
		* study the Q-function

		* then WE ADD A LONGITUDINAL CONTROLLERRRRR