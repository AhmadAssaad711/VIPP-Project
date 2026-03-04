### playing with the environment and reward function

* addressing the problem that agent memorized rather than generalized
	* physics priors in 'curvature' when tasked against new test envrionemtns did not know how to generalize and basic agent generalized better


		* so want to change a couple of things
		* curvature is currently map-centric defined, then agent would memorize rather than generalize

### memorization vs generalization




#### changing observation space

* including curvature change
* question is, how can curvature help training?

- different Q-table sizes, unfair....

### changing reward function

* an ablation study, how does each term affect the reward function
	* normal reward function: 3 terms norm error, alive rewad, and jerk

![[error_per_step_vs_training.png]]


![[path_following_q_learning.png]]





![[steps_vs_training.png]]

---
#### no alive reward:


![[error_per_step_vs_training 1.png]]

![[policy_heatmap.png]]


### jerk cost


- performance did not shift greatly, but it did shift nonetheless
- less steps per episode



## PIRL

![[Pasted image 20260304153042.png]]

- still even after changing the training data, bad performance
- but i think there is a fundemental error, that is with the constant lookahead distance, really is a big reason why the first formulation worked so much more better

---
### problem of curvature formulation

- scale mismatch:
	- normalization of curvature...
	- curvature error in reward function

## Physics-Informed Q-Learning Agent — Formulation Summary

### State Space

The state is a 4-tuple of discretized variables:

s=(ey, eψ, κnear, κla)s=(ey​, eψ​, κnear​, κla​)

|Variable|Description|Range|Bins|
|---|---|---|---|
|eyey​|Lateral error (signed distance from lane center)|[−2.0, 2.0][−2.0, 2.0] m|10|
|eψeψ​|Heading error (vehicle heading minus lane heading)|[−45°, 45°][−45°, 45°]|10|
|κnearκnear​|Lane curvature at nearest point (path property)|[−0.2, 0.2][−0.2, 0.2] 1/m|5|
|κlaκla​|Lane curvature at lookahead point (path property)|[−0.2, 0.2][−0.2, 0.2] 1/m|5|

**Total state space size:**

∣S∣=11×11×6×6=4,356∣S∣=11×11×6×6=4,356

(bin counts are N+1N+1 due to `np.digitize` producing indices in [0,N][0,N])

#### Curvature Estimation

Lane curvature is computed via finite differences of heading over an arc-length window:

κ(s)=ΔψΔs=ψlane(s+Δs)−ψlane(s)Δsκ(s)=ΔsΔψ​=Δsψlane​(s+Δs)−ψlane​(s)​

where Δs=2LΔs=2L (twice the car length).

- **Near curvature**: evaluated at the vehicle's arc-length coordinate ss
- **Lookahead curvature**: evaluated at s+dlas+dla​, where dla=5Ldla​=5L

> [!note] Design choice  
> Curvature features are **path properties only** — they inform the agent about upcoming geometry. The agent's own curvature (from finite-differenced heading) is **not** in the state, as it is noisy and redundant with the commanded action.

---

### Action Space

The agent selects a discrete curvature command κcmdκcmd​, converted to a steering angle via the bicycle model:

δ=arctan⁡(L⋅κcmd)δ=arctan(L⋅κcmd​)

where LL is the vehicle wheelbase (car length).

|Parameter|Value|
|---|---|
|Number of actions|Na=11Na​=11|
|Curvature range|κcmd∈[−0.2, 0.2]κcmd​∈[−0.2, 0.2] 1/m|
|Action set|A=linspace(−0.2, 0.2, 11)A=linspace(−0.2, 0.2, 11)|

---

### Reward Function

rt=ralive−E(ey, eψ)−λj⋅Jtrt​=ralive​−E(ey​, eψ​)−λj​⋅Jt​

where:

**Error term:**

E(ey, eψ)=(eyey,max⁡)2+λψ(eψeψ,max⁡)2E(ey​, eψ​)=(ey,max​ey​​)2+λψ​(eψ,max​eψ​​)2

**Jerk cost (steering smoothness):**

Jt=(κcmd,t−κcmd,t−1)2Jt​=(κcmd,t​−κcmd,t−1​)2

| Parameter                   | Symbol         | Value    |
| --------------------------- | -------------- | -------- |
| Alive reward                | raliveralive​  | 0.20.2   |
| Heading error weight        | λψλψ​          | 0.50.5   |
| Jerk penalty weight         | λjλj​          | 0.10.1   |
| Lateral error normalization | ey,max⁡ey,max​ | 2.02.0 m |
| Heading error normalization | eψ,max⁡eψ,max​ | 45°45°   |

> [!important] Reward design principles
> 
> - **Curvature is NOT penalized** in the reward — it is a path property, not an agent error
> - **Lateral and heading errors** are the only tracking penalties
> - **Jerk cost** encourages smooth steering transitions
> - **Alive reward** incentivizes survival (staying on-road)

---

### Q-Learning Update

Standard tabular Q-learning with ϵϵ-greedy exploration:

Q(st,at)←Q(st,at)+α[rt+γmax⁡a′Q(st+1,a′)−Q(st,at)]Q(st​,at​)←Q(st​,at​)+α[rt​+γa′max​Q(st+1​,a′)−Q(st​,at​)]

|Hyperparameter|Symbol|Value|
|---|---|---|
|Learning rate|αα|0.10.1|
|Discount factor|γγ|0.990.99|
|Initial exploration|ϵ0ϵ0​|1.01.0|
|Exploration decay|—|ϵ←max⁡(0.05, 0.99985⋅ϵ)ϵ←max(0.05, 0.99985⋅ϵ)|
|Training episodes|—|20,00020,000|
|Max steps per episode|—|200200|

---

### Training Setup

- **Multi-map curriculum**: 5 fixed track seeds, rotated every 100 episodes
- **Randomized tracks**: Straight lengths, arc radii, and sweep angles are randomized per seed
- **Evaluation**: 200 episodes on unseen random maps (greedy policy, no exploration)
- **Termination**: Episode ends if vehicle leaves the road or reaches 200 steps

---

### Architecture Diagram


┌─────────────────────────────────────────────┐
│                  STATE                       │
│  (e_y, e_ψ, κ_near, κ_la)                  │
│   ↑ tracking errors   ↑ path preview        │
└──────────────┬──────────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │   Q-Table    │  4,356 states × 11 actions
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  κ_cmd ∈ A   │  curvature command
        └──────┬───────┘
               │  δ = arctan(L · κ_cmd)
               ▼
        ┌──────────────┐
        │  Environment │  highway-env racetrack
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │    Reward     │  r = 0.2 - E(e_y, e_ψ) - λ_j · J
        └──────────────┘




==key ideas to look out for:==

* convergence of the Q-table, how much of the Q-table has been visited
* also want to do an ablation study for the parameters weights and terms
* 