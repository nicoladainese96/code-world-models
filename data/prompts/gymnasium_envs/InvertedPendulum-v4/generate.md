<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description

This environment involves a cart that can moved linearly, with a pole fixed on it
at one end and having another end free. The cart can be pushed left or right, and the
goal is to balance the pole on the top of the cart by applying forces on the cart.

## Action Space
The agent take a 1-element vector for actions.

The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
the numerical force applied to the cart (with magnitude representing the amount of
force and sign representing the direction)

| Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
|-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
| 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |

## Observation Space

The state space consists of positional values of different body parts of
the pendulum system, followed by the velocities of those individual parts (their derivatives)
with all the positions ordered before all the velocities.

The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:

| Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
| --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
| 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
| 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
| 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
| 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | anglular velocity (rad/s) |


## Rewards

The goal is to make the inverted pendulum stand upright (within a certain angle limit)
as long as possible - as such a reward of +1 is awarded for each timestep that
the pole is upright.

## Starting State
All observations start in state
(0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
of [-0.01, 0.01] added to the values for stochasticity.

## Episode End
The episode ends when any of the following happens:

1. Truncation: The episode duration reaches 1000 timesteps.
2. Termination: Any of the state space values is no longer finite.
3. Termination: The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radian.

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