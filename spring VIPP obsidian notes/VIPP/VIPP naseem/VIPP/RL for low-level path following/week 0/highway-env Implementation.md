problem formulation of path-following car using basic RL
* so the whole point is to minimize the error between where it is and where it should be
* first tried a wall-following approach, similar to basic algorithms, did not learn since there was no sense of how much should i turn
* then i tried using a kinematic approach (coordinate systems)

	* here there was the question of what kind of coordinates do i use, global or serret frenet, tried global good behaviour (idea with serret frenet is less variables)
	* now the way i structured the problem is (x,y,psi, then some information of where the path is relative to the vehicle)
	* actions it can take are the steering rate input
	* change in x and y are easy to calculate
	* cool about cartesian is that physics is easy

* the reward function involved 2 normalized terms, first is how bad am i with respect to where i need to be and second is jerk, how much am i steering

NEXT steps: 
* try serret frenet frame
* inject physics
* make a clear baseline and success criteria

