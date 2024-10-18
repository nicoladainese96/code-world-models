<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description

The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8
joints connecting them (including two paws). The goal is to apply a torque
on the joints to make the cheetah run forward (right) as fast as possible,
with a positive reward allocated based on the distance moved forward and a
negative reward allocated for moving backward. The torso and head of the
cheetah are fixed, and the torque can only be applied on the other 6 joints
over the front and back thighs (connecting to the torso), shins
(connecting to the thighs) and feet (connecting to the shins).

## Action Space
The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.

| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
| 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
| 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
| 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
| 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
| 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |


## Observation Space
Observations consist of positional values of different body parts of the
cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

By default, observations do not include the cheetah's `rootx`. It may
be included by passing `exclude_current_positions_from_observation=False` during construction.
In that case, the observation space will be a `Box(-Inf, Inf, (18,), float64)` where the first element
represents the `rootx`.
Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the
will be returned in `info` with key `"x_position"`.

However, by default, the observation is a `Box(-Inf, Inf, (17,), float64)` where the elements correspond to the following:

| Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
| --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
| 0   | z-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | position (m)             |
| 1   | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angle (rad)              |
| 2   | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
| 3   | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angle (rad)              |
| 4   | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
| 5   | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
| 6   | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angle (rad)              |
| 7   | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
| 8   | x-coordinate of the front tip        | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
| 9   | y-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
| 10  | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
| 11  | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
| 12  | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
| 13  | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
| 14  | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
| 15  | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
| 16  | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |
| excluded |  x-coordinate of the front tip  | -Inf | Inf | rootx                            | slide | position (m)             |

## Rewards
The reward consists of two parts:
- *forward_reward*: A reward of moving forward which is measured
as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
the time between actions and is dependent on the frame_skip parameter
(fixed to 5), where the frametime is 0.01 - making the
default *dt = 5 * 0.01 = 0.05*. This reward would be positive if the cheetah
runs forward (right).
- *ctrl_cost*: A cost for penalising the cheetah if it takes
actions that are too large. It is measured as *`ctrl_cost_weight` *
sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
control and has a default value of 0.1

The total reward returned is ***reward*** *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

## Starting State
All observations start in state (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,) with a noise added to the
initial state for stochasticity. As seen before, the first 8 values in the
state are positional and the last 9 values are velocity. A uniform noise in
the range of [-`reset_noise_scale`, `reset_noise_scale`] is added to the positional values while a standard
normal noise with a mean of 0 and standard deviation of `reset_noise_scale` is added to the
initial velocity values of all zeros.

## Episode End
The episode truncates when the episode length is greater than 1000.

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