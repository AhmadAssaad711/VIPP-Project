### looking into reward shaping and longitudinal controllers for RL

- when i add velocity, i need to also look at vehicle dynamics and how they interact together
- lateral and longitudinal dynamics depend on each other, friction


### motivation:
- currently model based controllers are great at modelling the neede velocity of the autonomous vehicle, however they struggle with estimating unknown vehicle dynamics and unaccounted for nonlinearities
	- a great entrance to research may be looking at the behaviour of a lateral + longitudinal controller under varying road conditions unnown to the agent
	- another idea, could be the idea to use RL as an optimization method in finding the shortest path along a track while respecting dynamics
	- 