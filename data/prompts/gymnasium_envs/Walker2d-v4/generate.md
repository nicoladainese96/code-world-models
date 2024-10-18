<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description

This environment builds on the hopper environment
by adding another set of legs making it possible for the robot to walk forward instead of
hop. Like other Mujoco environments, this environment aims to increase the number of independent state
and control variables as compared to the classic control environments. The walker is a
two-dimensional two-legged figure that consist of seven main body parts - a single torso at the top
(with the two legs splitting after the torso), two thighs in the middle below the torso, two legs
in the bottom below the thighs, and two feet attached to the legs on which the entire body rests.
The goal is to walk in the in the forward (right)
direction by applying torques on the six hinges connecting the seven body parts.

## Action Space
The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.

| Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
| 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
| 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
| 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
| 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
| 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

## Observation Space
Observations consist of positional values of different body parts of the walker,
followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

By default, observations do not include the x-coordinate of the torso. It may
be included by passing `exclude_current_positions_from_observation=False` during construction.
In that case, the observation space will be `Box(-Inf, Inf, (18,), float64)` where the first observation
represent the x-coordinates of the torso of the walker.
Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
of the torso will be returned in `info` with key `"x_position"`.

By default, observation is a `Box(-Inf, Inf, (17,), float64)` where the elements correspond to the following:

| Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
| --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
| excluded | x-coordinate of the torso                     | -Inf | Inf | rootx                            | slide | position (m)             |
| 0   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
| 1   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
| 2   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
| 3   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
| 4   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
| 5   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
| 6   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
| 7   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
| 8   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
| 9   | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
| 10  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
| 11  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
| 12  | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
| 13  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
| 14  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
| 15  | angular velocity of the leg hinge                  | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
| 16  | angular velocity of the foot hinge                 | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

## Rewards
The reward consists of three parts:
- *healthy_reward*: Every timestep that the walker is alive, it receives a fixed reward of value `healthy_reward`,
- *forward_reward*: A reward of walking forward which is measured as
*`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
*dt* is the time between actions and is dependeent on the frame_skip parameter
(default is 4), where the frametime is 0.002 - making the default
*dt = 4 * 0.002 = 0.008*. This reward would be positive if the walker walks forward (positive x direction).
- *ctrl_cost*: A cost for penalising the walker if it
takes actions that are too large. It is measured as
*`ctrl_cost_weight` * sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is
a parameter set for the control and has a default value of 0.001

The total reward returned is ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

## Starting State
All observations start in state
(0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
with a uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.

## Episode End
The walker is said to be unhealthy if any of the following happens:

1. Any of the state space values is no longer finite
2. The height of the walker is ***not*** in the closed interval specified by `healthy_z_range`
3. The absolute value of the angle (`observation[1]` if `exclude_current_positions_from_observation=False`, else `observation[2]`) is ***not*** in the closed interval specified by `healthy_angle_range`

If `terminate_when_unhealthy=True` is passed during construction (which is the default),
the episode ends when any of the following happens:

1. Truncation: The episode duration reaches a 1000 timesteps
2. Termination: The walker is unhealthy

If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.

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