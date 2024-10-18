<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description
"Reacher" is a two-jointed robot arm. The goal is to move the robot's end effector (called *fingertip*) close to a
target that is spawned at a random position.

## Action Space
The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

| Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
|-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
| 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1 | 1 | joint0  | hinge | torque (N m) |
| 1   |  Torque applied at the second hinge (connecting the two links)                  | -1 | 1 | joint1  | hinge | torque (N m) |

## Observation Space
Observations consist of

- The cosine of the angles of the two arms
- The sine of the angles of the two arms
- The coordinates of the target
- The angular velocities of the arms
- The vector between the target and the reacher's fingertip (3 dimensional with the last element being 0)

The observation is a `Box(-Inf, Inf, (11,), float64)` where the elements correspond to the following:

| Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
| --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
| 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                      | hinge | unitless                 |
| 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                      | hinge | unitless                 |
| 2   | sine of the angle of the first arm                                                             | -Inf | Inf | sin(joint0)                      | hinge | unitless                 |
| 3   | sine of the angle of the second arm                                                            | -Inf | Inf | sin(joint1)                      | hinge | unitless                 |
| 4   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                         | slide | position (m)             |
| 5   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                         | slide | position (m)             |
| 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                           | hinge | angular velocity (rad/s) |
| 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                           | hinge | angular velocity (rad/s) |
| 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
| 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
| 10  | z-value of position_fingertip - position_target (constantly 0 since reacher is 2d and z is same for both) | -Inf | Inf | NA                               | slide | position (m)             |


Most Gym environments just return the positions and velocity of the
joints in the `.xml` file as the state of the environment. However, in
reacher the state is created by combining only certain elements of the
position and velocity, and performing some function transformations on them.
If one is to read the `.xml` for reacher then they will find 4 joints:

| Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
|-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
| 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
| 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
| 2   | x-coordinate of the target  | -Inf     | Inf      | target_x                         | slide | position (m)       |
| 3   | y-coordinate of the target  | -Inf     | Inf      | target_y                         | slide | position (m)       |


## Rewards
The reward consists of two parts:
- *reward_distance*: This reward is a measure of how far the *fingertip*
of the reacher (the unattached end) is from the target, with a more negative
value assigned for when the reacher's *fingertip* is further away from the
target. It is calculated as the negative vector norm of (position of
the fingertip - position of target), or *-norm("fingertip" - "target")*.
- *reward_control*: A negative reward for penalising the walker if
it takes actions that are too large. It is measured as the negative squared
Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

The total reward returned is ***reward*** *=* *reward_distance + reward_control*

Unlike other environments, Reacher does not allow you to specify weights for the individual reward terms.
However, `info` does contain the keys *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms,
you should create a wrapper that computes the weighted reward from `info`.


## Starting State
All observations start in state
(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
with a noise added for stochasticity. A uniform noise in the range
[-0.1, 0.1] is added to the positional attributes, while the target position
is selected uniformly at random in a disk of radius 0.2 around the origin.
Independent, uniform noise in the
range of [-0.005, 0.005] is added to the velocities, and the last
element ("fingertip" - "target") is calculated at the end once everything
is set. The default setting has a framerate of 2 and a *dt = 2 * 0.01 = 0.02*

## Episode End

The episode ends when any of the following happens:

1. Truncation: The episode duration reaches a 50 timesteps (with a new random target popping up if the reacher's fingertip reaches it before 50 timesteps)
2. Termination: Any of the state space values is no longer finite.

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