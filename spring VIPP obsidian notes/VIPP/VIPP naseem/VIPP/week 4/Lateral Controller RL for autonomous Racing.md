* i should go back to my environment, play with code alittle
	* need to create a variable controller, try out a PID
	* goal should be minimizing time....?
	* path following... how can i ensure that it is also racing
	* so play with code
		* generalizability, check that out
		* curvature bins...?

---
#### for next time:
* play with environment, formulate why it didnt learn what I wanted
* try inserting a  longitudinal controller
	* introduce metrics to benchmark at

--- 
lit review: 

* *reward shaping*
* *longitudinal controller* 



### playing with the environment and reward function

* addressing the problem that agent memorized rather than generalized
	* physics priors in 'curvature' when tasked against new test envrionemtns did not know how to generalize and basic agent generalized better


		* so want to change a couple of things
		* curvature is currently map-centric defined, then agent would memorize rather than generalize
		* 