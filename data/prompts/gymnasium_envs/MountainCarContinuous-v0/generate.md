<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description

The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
that can be applied to the car in either direction. The goal of the MDP is to strategically
accelerate the car to reach the goal state on top of the right hill. There are two versions
of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
This version is the one with continuous actions.

## Observation Space

The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

| Num | Observation                          | Min  | Max | Unit         |
|-----|--------------------------------------|------|-----|--------------|
| 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
| 1   | velocity of the car                  | -Inf | Inf | position (m) |

## Action Space

The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.

## Reward

A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
is added to the negative reward for that timestep.

## Starting State

The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
The starting velocity of the car is always assigned to 0.

## Episode End

The episode ends if either of the following happens:
1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
2. Truncation: The length of the episode is 999.

## Class Definition
The class should be called "Environment". It should have at least:

- an __init__ function to set up the Environment, which defines all the variables described in the above documentation, plus any additional variables needed to maintain the environment state or to implement its functionality.
- a set_state function to set a custom value for the environment and its internal representation (you can assume that when "set_state" is used, the task is not done and internal variables should be set as a consequence). set_state takes a single argument as input: a state observation from the observation space defined above.
- a step function to predict a step in the environment. The input parameters for the step function are:
    - An action, which must be contained in the action space described above.
  
    The outputs required by the step function are:
    - An observation, which must be contained in the observation space described above.
    - The reward for taking the action, as described in the reward definition above.
    - A boolean variable indicating if the episode is done.

## Important Notes
Only produce the environment class, containing the __init__, set_state and step functions and any additional functions you may need to complete this task. Do not write an example of how to use the class or anything else.
Be careful about edge cases.
Make sure to write all the required functions and that they have the exact names as specified in the task description. Missing or incorrectly named functions will not pass the tests and will result in a score of 0.
It is of VITAL importance that you do not leave undefined any function, but implement each of them completely.
</user>

<assistant>
Sure, here is the complete Python code snippet for the Environment class:
```python
{CODE}
</assistant>