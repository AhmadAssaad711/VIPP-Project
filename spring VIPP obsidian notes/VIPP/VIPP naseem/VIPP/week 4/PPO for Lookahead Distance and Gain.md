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


Logger:
* compared generalization between agents on randomzied track, underwhelming results
* pre-read paper, will deep dive later at night
* read about PPO, TRPO, VPG, MPC, Residual RL




Paper Review:
***Goal of Paper is to create a PPO algorithm that outputs lookahead distance (Ld) and steering gain (g), again this is policy-guided parameter tuning ****

	what is the difference from MRAC if it is just simple parameter updating?
	what is the difference between such a method and using residual RL, where RL outputs its own control law?


==From Introduction:==

**MPC: i have model of system, look at N timesteps, choose action, then repeat all over again***
* i dont want to let go of this information that i gain, when i know the model MPC should be the way to go...?
* We formulate parameter sequencing as a continuous control problem and train with Proximal Policy Optimization (PPO) [2] to map compact state features—vehicle speed and curvature taps at near/mid/far horizons
* A first order smoother mitigates sudden action changes, and a safety “teacher” (linear v→Ld and v→g) serves as a fallback if RL commands become stale.
* **ablation method**: removing a component to see its effect, measuring performance change by removing an element and testing performance

===Literature Review===
* cool papers 12-17 on deep learning methods for control and RL interpratability
* main contribution is that instead of manually tuning for different parameters, learns on just this one
	* PP fixed and intrepretable steering law?
	* learning the ld and gain based on features


==Method==
For the global reference, we use TUM’s global_racetrajectory_optimization [4] to compute a smooth minimum-curvature raceline within track boundaries. The optimizer returns a closed-loop trajectory with associated curvature κ(s) and a friction limited speed profile vmax(s), which we export as waypoints {x(s),y(s),κ(s),vmax(s)}. This globally smooth geometry provides both the tracking reference and curvature features used by the controller and the RL policy.
	* minimize curvature -> minimize steering effort
	* optimizer not only returns the centerline but returns the minimum effort centerline
	* parameterized by arc-length and not time
	* tracking reference and curvature features to be used by the controller and the policy

Hierarchal RL instead of end-to-end RL


==Observation and Action Spaces===
$$
s_t =
\begin{bmatrix}
v_t \\
\kappa_{0,t} \\
\kappa_{1,t} \\
\kappa_{2,t} \\
\Delta \kappa_t
\end{bmatrix},
\qquad
\Delta \kappa_t = \kappa_{1,t} - \kappa_{0,t}
$$

$$
a_t =
\begin{bmatrix}
L_{t+1} \\
g_{t+1}
\end{bmatrix},
\qquad
L_{t+1} \in [0.35,\,4.0], \quad
g_{t+1} \in [0.45,\,1.15]
$$

$$
\tilde{L}_{t+1}
=
\beta_L L_{t+1}
+
(1-\beta_L)\tilde{L}_t
$$

$$
\tilde{g}_{t+1}
=
\beta_g g_{t+1}
+
(1-\beta_g)\tilde{g}_t
$$

state space, action space, bounds for limits
* the curvatures are for low mid and high.... actually highlights why my formulation was partially mistaken (if not completely)
* they are then smoothed:
	* purpose is to eliminate jumping between actions (instead of just penalizing it in he reward function)


#### observation space

for the curvatures, basically go to nearest waypoint on track, ==offset by a set parameter (0, 5, 12)==
* what about a varying state? 
	* but that asks about the convergence and performance of the different RL algorithms


#### reward function
$$
R_t = w_v v_t
- w_L \left| \tilde{L}_t - L_t^* \right|
- w_G \left| \tilde{g}_t - g_t^* \right|
- w_{jL} \left| \tilde{L}_t - \tilde{L}_{t-1} \right|
- w_{jG} \left| \tilde{g}_t - \tilde{g}_{t-1} \right|
- w_k |\kappa_t|
- w_\times \left( \tilde{L}_t \kappa_t^{\max} \right)
+ w_{\mathrm{pre}} \,\mathbb{I}_{\mathrm{bend}}(\kappa_t^{\max})
  \,\mathbb{I}\!\left[\tilde{L}_t \le \ell(v_t)\right]
- w_c \,\mathbb{I}_{\mathrm{collision}}
- w_s \,\mathbb{I}_{\mathrm{slow}}
+ w_p \,\Delta p_t
$$

$$
\kappa_t^{\max} = \max(\kappa_{0,t}, \kappa_{1,t}, \kappa_{2,t})
$$

$$
L_t^* = \mathrm{clip}\!\left(0.50 + 0.28 v_t - 3.5 \kappa_t^{\max},\, 0.35,\, 4.0\right)
$$
$$
g_t^* = \mathrm{clip}\!\left(m v_t + b,\, 0.45,\, 1.15\right)
$$

==**steering gain is a measure of how much I should follow the steering angle==

so, they separated throttle from steering
* what is the idea behind keeping throttle and controlling it using RL?


### Why $L_d$ and $g$ are Non-Redundant

The commanded curvature is approximately

$$
\gamma = \frac{2 g\, y'}{L_d^2}
$$

where $y'$ depends on the preview target point, and the target point depends on $L_d$.

---

**Effect of $g$:**

$$
\frac{\partial \gamma}{\partial g}
=
\frac{2 y'}{L_d^2}
$$

Changing $g$ only scales the curvature.  
It does **not** move the target point.  
Thus, $g$ acts as a pure multiplier.

---

**Effect of $L_d$:**

Using the chain rule,

$$
\frac{d\gamma}{dL_d}
=
-\frac{4 g y'}{L_d^3}
+
\frac{2g}{L_d^2}\frac{d y'}{dL_d}
$$

The first term is explicit scaling.  
The second term arises because $y'$ depends on $L_d$ (the preview point moves).

Thus, changing $L_d$:
- rescales curvature, **and**
- changes the preview geometry.

---

**Conclusion**

$g$ changes magnitude only.  
$L_d$ changes both magnitude and geometry.  

Therefore, $L_d$ and $g$ are structurally non-redundant.


### PPO

#### Big picture

Each training cycle looks like:

1. Collect 4096 steps (on-policy).
    
2. Compute advantages.
    
3. Shuffle data.
    
4. Run 5 optimization passes over it.
    
5. Update policy.
    
6. Repeat.


### **nsteps**

Number of environment time steps collected **before one policy update**.

### **Minibatch size**

Number of samples used in **one gradient step** during optimization.


### **nepochs**

Number of times you pass over the same collected rollout data.



### Algorithm details


Observations and returns are normalized on
line using $\mathrm{VecNormalize}$ for stability.

PPO maximizes the clipped surrogate objective:

$$
L_{\text{clip}}(\theta)
=
\mathbb{E}_t \Big[
\min \big(
r_t(\theta)\,\hat{A}_t,\;
\operatorname{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t
\big)
\Big]
$$

where

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

and $\hat{A}_t$ denotes the advantage estimate.

The full PPO objective combines policy, value, and entropy terms:

$$
L_{\text{PPO}}(\theta,\phi)
=
- L_{\text{clip}}(\theta)
+ c_v \,\mathbb{E}_t \big( V_\phi(s_t) - \hat{R}_t \big)^2
- c_s \,\mathbb{E}_t \mathcal{H}\big(\pi_\theta(\cdot \mid s_t)\big)
$$

with

$$
c_v = 0.6,
\qquad
c_s = 0.02.
$$






### Comparision of different algorithms

### The key insight

MPC is optimal **given its model, horizon, and cost**.

If any of those are imperfect, performance is limited.

RL is not more powerful in theory.

It is just more flexible when:

- The model is wrong
    
- The cost is misspecified
    
- The horizon is too short
    
- The system is highly nonlinear


**MPC is finite horizion, cannot account for future rewards beyond this frame of time**



### Pure pursuit

Highly dependent on the velocity speed and pre-chosen lookahead distance...



Yes cool, but constant velocity...., hyperparameters in states....., ablation study on lookahead and steering gain
