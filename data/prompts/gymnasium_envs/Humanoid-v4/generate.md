<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description

The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen) with a pair of
legs and arms. The legs each consist of three body parts, and the arms 2 body parts (representing the knees and
elbows respectively). The goal of the environment is to walk forward as fast as possible without falling over.

## Action Space
The action space is a `Box(-1, 1, (17,), float32)`. An action represents the torques applied at the hinge joints.

| Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
|-----|----------------------|---------------|----------------|---------------------------------------|-------|------|
| 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4 | 0.4 | abdomen_y                   | hinge | torque (N m) |
| 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4 | 0.4 | abdomen_z                   | hinge | torque (N m) |
| 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4 | 0.4 | abdomen_x                   | hinge | torque (N m) |
| 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4 | 0.4 | right_hip_x (right_thigh)   | hinge | torque (N m) |
| 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4 | 0.4 | right_hip_z (right_thigh)   | hinge | torque (N m) |
| 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4 | 0.4 | right_hip_y (right_thigh)   | hinge | torque (N m) |
| 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4 | 0.4 | right_knee                  | hinge | torque (N m) |
| 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4 | 0.4 | left_hip_x (left_thigh)     | hinge | torque (N m) |
| 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4 | 0.4 | left_hip_z (left_thigh)     | hinge | torque (N m) |
| 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4 | 0.4 | left_hip_y (left_thigh)     | hinge | torque (N m) |
| 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4 | 0.4 | left_knee                   | hinge | torque (N m) |
| 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4 | 0.4 | right_shoulder1             | hinge | torque (N m) |
| 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4 | 0.4 | right_shoulder2             | hinge | torque (N m) |
| 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4 | 0.4 | right_elbow                 | hinge | torque (N m) |
| 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4 | 0.4 | left_shoulder1              | hinge | torque (N m) |
| 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4 | 0.4 | left_shoulder2              | hinge | torque (N m) |
| 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4 | 0.4 | left_elbow                  | hinge | torque (N m) |

## Observation Space
Observations consist of positional values of different body parts of the Humanoid,
followed by the velocities of those individual parts (their derivatives) with all the
positions ordered before all the velocities.

By default, observations do not include the x- and y-coordinates of the torso. These may
be included by passing `exclude_current_positions_from_observation=False` during construction.
In that case, the observation space will be a `Box(-Inf, Inf, (378,), float64)` where the first two observations
represent the x- and y-coordinates of the torso.
Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x- and y-coordinates
will be returned in `info` with keys `"x_position"` and `"y_position"`, respectively.

However, by default, the observation is a `Box(-Inf, Inf, (376,), float64)`. The elements correspond to the following:

| Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Unit                       |
| --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
| 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
| 1   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 2   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 3   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 4   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_z                        | hinge | angle (rad)                |
| 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_y                        | hinge | angle (rad)                |
| 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)                |
| 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_x                      | hinge | angle (rad)                |
| 9   | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_z                      | hinge | angle (rad)                |
| 10  | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_y                      | hinge | angle (rad)                |
| 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)                |
| 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_x                       | hinge | angle (rad)                |
| 13  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_z                       | hinge | angle (rad)                |
| 14  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_y                       | hinge | angle (rad)                |
| 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)                |
| 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder1                  | hinge | angle (rad)                |
| 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder2                  | hinge | angle (rad)                |
| 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)                |
| 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder1                   | hinge | angle (rad)                |
| 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder2                   | hinge | angle (rad)                |
| 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)                |
| 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
| 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
| 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
| 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | anglular velocity (rad/s)  |
| 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | anglular velocity (rad/s)  |
| 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | aanglular velocity (rad/s) |
| 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_x                      | hinge | anglular velocity (rad/s)  |
| 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | anglular velocity (rad/s)  |
| 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | anglular velocity (rad/s)  |
| 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | anglular velocity (rad/s)  |
| 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_x                       | hinge | anglular velocity (rad/s)  |
| 36  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | anglular velocity (rad/s)  |
| 37  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | anglular velocity (rad/s)  |
| 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | anglular velocity (rad/s)  |
| 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder1                  | hinge | anglular velocity (rad/s)  |
| 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder2                  | hinge | anglular velocity (rad/s)  |
| 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | anglular velocity (rad/s)  |
| 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder1                   | hinge | anglular velocity (rad/s)  |
| 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder2                   | hinge | anglular velocity (rad/s)  |
| 44  | angular velocity of the angle between left upper arm and left_lower_arm                                         | -Inf | Inf | left_elbow                       | hinge | anglular velocity (rad/s)  |
| excluded | x-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |
| excluded | y-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |

Additionally, after all the positional and velocity based values in the table,
the observation contains (in order):
- *cinert:* Mass and inertia of a single rigid body relative to the center of mass
(this is an intermediate result of transition). It has shape 14*10 (*nbody * 10*)
and hence adds to another 140 elements in the state space.
- *cvel:* Center of mass based velocity. It has shape 14 * 6 (*nbody * 6*) and hence
adds another 84 elements in the state space
- *qfrc_actuator:* Constraint force generated as the actuator force. This has shape
`(23,)`  *(nv * 1)* and hence adds another 23 elements to the state space.
- *cfrc_ext:* This is the center of mass based external force on the body.  It has shape
14 * 6 (*nbody * 6*) and hence adds to another 84 elements in the state space.
where *nbody* stands for the number of bodies in the robot and *nv* stands for the
number of degrees of freedom (*= dim(qvel)*)

The body parts are:

| id (for `v2`,`v3`,`v4`) | body part |
| --- |  ------------  |
| 0   | worldBody (note: all values are constant 0) |
| 1   | torso |
| 2   | lwaist |
| 3   | pelvis |
| 4   | right_thigh |
| 5   | right_sin |
| 6   | right_foot |
| 7   | left_thigh |
| 8   | left_sin |
| 9   | left_foot |
| 10  | right_upper_arm |
| 11  | right_lower_arm |
| 12  | left_upper_arm |
| 13  | left_lower_arm |

The joints are:

| id (for `v2`,`v3`,`v4`) | joint |
| --- |  ------------  |
| 0   | root |
| 1   | root |
| 2   | root |
| 3   | root |
| 4   | root |
| 5   | root |
| 6   | abdomen_z |
| 7   | abdomen_y |
| 8   | abdomen_x |
| 9   | right_hip_x |
| 10  | right_hip_z |
| 11  | right_hip_y |
| 12  | right_knee |
| 13  | left_hip_x |
| 14  | left_hiz_z |
| 15  | left_hip_y |
| 16  | left_knee |
| 17  | right_shoulder1 |
| 18  | right_shoulder2 |
| 19  | right_elbow|
| 20  | left_shoulder1 |
| 21  | left_shoulder2 |
| 22  | left_elfbow |

The (x,y,z) coordinates are translational DOFs while the orientations are rotational
DOFs expressed as quaternions. One can read more about free joints on the
[Mujoco Documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

**Note:** Humanoid-v4 environment no longer has the following contact forces issue.
If using previous Humanoid versions from v4, there have been reported issues that using a Mujoco-Py version > 2.0
results in the contact forces always being 0. As such we recommend to use a Mujoco-Py
version < 2.0 when using the Humanoid environment if you would like to report results
with contact forces (if contact forces are not used in your experiments, you can use
version > 2.0).

## Rewards
The reward consists of three parts:
- *healthy_reward*: Every timestep that the humanoid is alive (see section Episode Termination for definition), it gets a reward of fixed value `healthy_reward`
- *forward_reward*: A reward of walking forward which is measured as *`forward_reward_weight` *
(average center of mass before action - average center of mass after action)/dt*.
*dt* is the time between actions and is dependent on the frame_skip parameter
(default is 5), where the frametime is 0.003 - making the default *dt = 5 * 0.003 = 0.015*.
This reward would be positive if the humanoid walks forward (in positive x-direction). The calculation
for the center of mass is defined in the `.py` file for the Humanoid.
- *ctrl_cost*: A negative reward for penalising the humanoid if it has too
large of a control force. If there are *nu* actuators/controls, then the control has
shape  `nu x 1`. It is measured as *`ctrl_cost_weight` * sum(control<sup>2</sup>)*.
- *contact_cost*: A negative reward for penalising the humanoid if the external
contact force is too large. It is calculated by clipping
*`contact_cost_weight` * sum(external contact force<sup>2</sup>)* to the interval specified by `contact_cost_range`.

The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost* and `info` will also contain the individual reward terms

## Starting State
All observations start in state
(0.0, 0.0,  1.4, 1.0, 0.0  ... 0.0) with a uniform noise in the range
of [-`reset_noise_scale`, `reset_noise_scale`] added to the positional and velocity values (values in the table)
for stochasticity. Note that the initial z coordinate is intentionally
selected to be high, thereby indicating a standing up humanoid. The initial
orientation is designed to make it face forward as well.

## Episode End
The humanoid is said to be unhealthy if the z-position of the torso is no longer contained in the
closed interval specified by the argument `healthy_z_range`.

If `terminate_when_unhealthy=True` is passed during construction (which is the default),
the episode ends when any of the following happens:

1. Truncation: The episode duration reaches a 1000 timesteps
3. Termination: The humanoid is unhealthy

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