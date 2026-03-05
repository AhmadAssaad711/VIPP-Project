- main idea is to use RL to learn how deal with complexity of environment
	- i think good ideas may be to take different scenarios of local path planning diverging from global path planning
	- then again, what is the big picture of my work


- main issue i am tackling is 
	- follow this path, if unable learn how to workaround it then follow path again
	- i think best environment may be highway env and no longer the racetrack environment
	- but different repo...???
	- i do want to continue working on the low-level

---
# RL for High-Level Decision Making in Path Following

## Motivation

The current work applies Reinforcement Learning (RL) for **low-level vehicle control**, where the agent directly outputs control commands such as steering and velocity.

To increase the complexity and scope of the project, the RL formulation is extended to **high-level decision making for local path planning**.

Instead of controlling actuators directly, the RL agent will determine **behavioral decisions when the nominal reference path becomes obstructed**.

Example scenario:

- The vehicle follows a reference path.
- A **static obstacle** appears on the path.
- The RL agent must decide **how the vehicle should react**.

Possible behaviors include:

- Continue following the path
- Slow down
- Stop
- Deviate from the path to avoid the obstacle
- Select an alternative local trajectory around the obstacle

---

## Problem Formulation

The goal is to design an RL agent capable of **making local planning decisions** during path following.

The agent operates within a **hierarchical control architecture**:

---


