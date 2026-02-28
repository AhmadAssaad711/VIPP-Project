* neural networks learning is a relative blackbox, the patterns formed from the math is abstract
* use of shapley values for local insights rather than global, reproducible understanding of policy behaviour
* **paper: From Explainability to Interpretability: Interpretable Policies in Reinforcement Learning Via Model Explanation
	* use of a model-agnostic method to explain deep RL policies
	* shapley values
	* can i apply this here...??
* voronoi vectors
* vector quantization
* graph trees

**Rule-Guided Reinforcement Learning Policy Evaluation and Improvement Martin Tappler1 , Ignacio D. Lopez-Miguel1 , Sebastian Tschiatschek2 and Ezio Bartocci1 1TU Wien 2University of Vienna**


* using metamorphic relations to test a learned policy, metamorphic relations include symmetry such as rotation
	* basically creating structure where the agent can learn from 'similar' feature states

they mine rules, generalize through symmetry and relations in environments, and finally rule guided execution

we want learning methods to lead to rule based, or at least something we can generalize to

now the process of how to generalize symmetries:
	Step 1:
		*  rule mining as a classification problem: we have a map of all (s,a) pairs, we want to learn actions in certain states, must be accurate and have high coverage
		* they invent a mathematical form for this
			* include +-, action, and feature importance (how much factors affected the agent's decision)
	Step 2:
		* whole goal is to learn from symmetrically equivalent scanrios, they define feature relations for this
			* they define the rules for states being related to feature relation which involves conjunctions in actions and states
	Step 3:
		* guiding the agent with these rules;


Anyways, my work would focus much more on using RL for control purposes....



