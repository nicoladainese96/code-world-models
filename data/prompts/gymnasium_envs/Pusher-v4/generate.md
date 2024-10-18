<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description
"Pusher" is a multi-jointed robot arm which is very similar to that of a human.
 The goal is to move a target cylinder (called *object*) to a goal position using the robot's end effector (called *fingertip*).
  The robot consists of shoulder, elbow, forearm, and wrist joints.

## Action Space
The action space is a `Box(-2, 2, (7,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

| Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0    | Rotation of the panning the shoulder                              | -2          | 2           | r_shoulder_pan_joint             | hinge | torque (N m) |
| 1    | Rotation of the shoulder lifting joint                            | -2          | 2           | r_shoulder_lift_joint            | hinge | torque (N m) |
| 2    | Rotation of the shoulder rolling joint                            | -2          | 2           | r_upper_arm_roll_joint           | hinge | torque (N m) |
| 3    | Rotation of hinge joint that flexed the elbow                     | -2          | 2           | r_elbow_flex_joint               | hinge | torque (N m) |
| 4    | Rotation of hinge that rolls the forearm                          | -2          | 2           | r_forearm_roll_joint             | hinge | torque (N m) |
| 5    | Rotation of flexing the wrist                                     | -2          | 2           | r_wrist_flex_joint               | hinge | torque (N m) |
| 6    | Rotation of rolling the wrist                                     | -2          | 2           | r_wrist_roll_joint               | hinge | torque (N m) |

## Observation Space

Observations consist of

- Angle of rotational joints on the pusher
- Angular velocities of rotational joints on the pusher
- The coordinates of the fingertip of the pusher
- The coordinates of the object to be moved
- The coordinates of the goal position

The observation is a `Box(-Inf, Inf, (23,), float64)` where the elements correspond to the table below.
An analogy can be drawn to a human arm in order to help understand the state space, with the words flex and roll meaning the
same as human joints.

| Num | Observation                                              | Min  | Max | Name (in corresponding XML file) | Joint    | Unit                     |
| --- | -------------------------------------------------------- | ---- | --- | -------------------------------- | -------- | ------------------------ |
| 0   | Rotation of the panning the shoulder                     | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angle (rad)              |
| 1   | Rotation of the shoulder lifting joint                   | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angle (rad)              |
| 2   | Rotation of the shoulder rolling joint                   | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angle (rad)              |
| 3   | Rotation of hinge joint that flexed the elbow            | -Inf | Inf | r_elbow_flex_joint               | hinge    | angle (rad)              |
| 4   | Rotation of hinge that rolls the forearm                 | -Inf | Inf | r_forearm_roll_joint             | hinge    | angle (rad)              |
| 5   | Rotation of flexing the wrist                            | -Inf | Inf | r_wrist_flex_joint               | hinge    | angle (rad)              |
| 6   | Rotation of rolling the wrist                            | -Inf | Inf | r_wrist_roll_joint               | hinge    | angle (rad)              |
| 7   | Rotational velocity of the panning the shoulder          | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angular velocity (rad/s) |
| 8   | Rotational velocity of the shoulder lifting joint        | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angular velocity (rad/s) |
| 9   | Rotational velocity of the shoulder rolling joint        | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angular velocity (rad/s) |
| 10  | Rotational velocity of hinge joint that flexed the elbow | -Inf | Inf | r_elbow_flex_joint               | hinge    | angular velocity (rad/s) |
| 11  | Rotational velocity of hinge that rolls the forearm      | -Inf | Inf | r_forearm_roll_joint             | hinge    | angular velocity (rad/s) |
| 12  | Rotational velocity of flexing the wrist                 | -Inf | Inf | r_wrist_flex_joint               | hinge    | angular velocity (rad/s) |
| 13  | Rotational velocity of rolling the wrist                 | -Inf | Inf | r_wrist_roll_joint               | hinge    | angular velocity (rad/s) |
| 14  | x-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
| 15  | y-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
| 16  | z-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
| 17  | x-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidex)              | slide    | position (m)             |
| 18  | y-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidey)              | slide    | position (m)             |
| 19  | z-coordinate of the object to be moved                   | -Inf | Inf | object                           | cylinder | position (m)             |
| 20  | x-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidex)               | slide    | position (m)             |
| 21  | y-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidey)               | slide    | position (m)             |
| 22  | z-coordinate of the goal position of the object          | -Inf | Inf | goal                             | sphere   | position (m)             |


## Rewards
The reward consists of two parts:
- *reward_near *: This reward is a measure of how far the *fingertip*
of the pusher (the unattached end) is from the object, with a more negative
value assigned for when the pusher's *fingertip* is further away from the
target. It is calculated as the negative vector norm of (position of
the fingertip - position of target), or *-norm("fingertip" - "target")*.
- *reward_dist *: This reward is a measure of how far the object is from
the target goal position, with a more negative value assigned for object is
further away from the target. It is calculated as the negative vector norm of
(position of the object - position of goal), or *-norm("object" - "target")*.
- *reward_control*: A negative reward for penalising the pusher if
it takes actions that are too large. It is measured as the negative squared
Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

The total reward returned is ***reward*** *=* *reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near*

Unlike other environments, Pusher does not allow you to specify weights for the individual reward terms.
However, `info` does contain the keys *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms,
you should create a wrapper that computes the weighted reward from `info`.


## Starting State
All pusher (not including object and goal) states start in
(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0). A uniform noise in the range
[-0.005, 0.005] is added to the velocity attributes only. The velocities of
the object and goal are permanently set to 0. The object's x-position is selected uniformly
between [-0.3, 0] while the y-position is selected uniformly between [-0.2, 0.2], and this
process is repeated until the vector norm between the object's (x,y) position and origin is not greater
than 0.17. The goal always have the same position of (0.45, -0.05, -0.323).

The default framerate is 5 with each frame lasting for 0.01, giving rise to a *dt = 5 * 0.01 = 0.05*

## Episode End

The episode ends when any of the following happens:

1. Truncation: The episode duration reaches a 100 timesteps.
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