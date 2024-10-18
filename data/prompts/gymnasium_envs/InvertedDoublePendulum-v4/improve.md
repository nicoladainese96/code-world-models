<system>
You are an experienced Python developer. You will be provided with an incorrect code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to rewrite the program so that it performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description

This environment involves a cart that can
moved linearly, with a pole fixed on it and a second pole fixed on the other end of the first one
(leaving the second pole as the only one with one free end). The cart can be pushed left or right,
and the goal is to balance the second pole on top of the first pole, which is in turn on top of the
cart, by applying continuous forces on the cart.

## Action Space
The agent take a 1-element vector for actions.
The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
numerical force applied to the cart (with magnitude representing the amount of force and
sign representing the direction)

| Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
|-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
| 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |

## Observation Space

The state space consists of positional values of different body parts of the pendulum system,
followed by the velocities of those individual parts (their derivatives) with all the
positions ordered before all the velocities.

The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:

| Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
| --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
| 0   | position of the cart along the linear surface                     | -Inf | Inf | slider                           | slide | position (m)             |
| 1   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                       | hinge | unitless                 |
| 2   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                      | hinge | unitless                 |
| 3   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                       | hinge | unitless                 |
| 4   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                      | hinge | unitless                 |
| 5   | velocity of the cart                                              | -Inf | Inf | slider                           | slide | velocity (m/s)           |
| 6   | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s) |
| 7   | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                           | hinge | angular velocity (rad/s) |
| 8   | constraint force - 1                                              | -Inf | Inf |                                  |       | Force (N)                |
| 9   | constraint force - 2                                              | -Inf | Inf |                                  |       | Force (N)                |
| 10  | constraint force - 3                                              | -Inf | Inf |                                  |       | Force (N)                |


There is physical contact between the robots and their environment - and Mujoco
attempts at getting realistic physics simulations for the possible physical contact
dynamics by aiming for physical accuracy and computational efficiency.

There is one constraint force for contacts for each degree of freedom (3).
The approach and handling of constraints by Mujoco is unique to the simulator
and is based on their research. Once can find more information in their
[*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html)
or in their paper
["Analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo"](https://homes.cs.washington.edu/~todorov/papers/TodorovICRA14.pdf).


## Rewards

The reward consists of two parts:
- *alive_bonus*: The goal is to make the second inverted pendulum stand upright
(within a certain angle limit) as long as possible - as such a reward of +10 is awarded
 for each timestep that the second pole is upright.
- *distance_penalty*: This reward is a measure of how far the *tip* of the second pendulum
(the only free end) moves, and it is calculated as
*0.01 * x<sup>2</sup> + (y - 2)<sup>2</sup>*, where *x* is the x-coordinate of the tip
and *y* is the y-coordinate of the tip of the second pole.
- *velocity_penalty*: A negative reward for penalising the agent if it moves too
fast *0.001 *  v<sub>1</sub><sup>2</sup> + 0.005 * v<sub>2</sub> <sup>2</sup>*

The total reward returned is ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*

## Starting State
All observations start in state
(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
of [-0.1, 0.1] added to the positional values (cart position and pole angles) and standard
normal force with a standard deviation of 0.1 added to the velocity values for stochasticity.

## Episode End
The episode ends when any of the following happens:

1.Truncation:  The episode duration reaches 1000 timesteps.
2.Termination: Any of the state space values is no longer finite.
3.Termination: The y_coordinate of the tip of the second pole *is less than or equal* to 1. The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other).

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

First, write an explanation of the difference between the ground-truth transition and the step function's outputs in the example provided.
Second, point out the part of the code responsible for the incorrect prediction and why its logic is erroneous.
Third, suggest a concrete, actionable fix for it. 
Finally, fix the program in its entirety following the suggestion. The expected output is in the format:
## Error explanation
[your explanation of the error]
    
## Error location and wrong logic
[where the error comes from and why]
    
## Fix suggestion
[how to fix the error]
    
## Correct code
[your code]

## Incorrect code
You are provided with the following code snippet to fix.
```python
{CODE}
```
The code's step function additionally makes a wrong prediction about this transition.
## Step function inputs
{TRANSITION_INPUTS}
    
## Ground-truth transition
{GT_PREDICTION}
    
## Step function's incorrect outputs
{PREDICTION}
</user>