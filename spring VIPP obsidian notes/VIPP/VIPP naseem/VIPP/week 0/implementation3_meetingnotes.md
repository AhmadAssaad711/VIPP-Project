* tried to implement the problem on 3 levels
	* main goal is to follow the path, minimize position error

* first flavor:
	* observation space is only position error, nothing else
	* reward function includes position error and jerk and survival
	* key takeaways: does learn, slow learning, but a key problem with this is that it isnt markovian, 2 different states have the same meaning while they arent the same, key thing to point out in the heatmap
	* same state wrong action
* second flavor:
	* observation space includes position and heading errors, same reward function
	* key takeaways: 
		* observation space is more complex
		* notice that action space is still wrong and we notice it learns to 'cut corners'
		* faster learning, which is logical learns a measure of how much to turn in a sense, also markovian
* third flavor:
	* added curvature
	* it did learn, much slower, much more complexity
	* the idea behind curvature is that it should learn a measure of how much it should steer
	* but it didnt
next steps:
* inject more physics
* changing action space to include curvature rather than angle, it mimics the way pure pursuit works
* low level / high level controllers