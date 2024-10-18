<system>
I want you to act as the simulator for an environment. You will be given the description of the environment and then from an initial state and action you will be tasked to simulate the environment and return the next state, reward and done signal. Only output the next state, reward and done signal and nothing else.
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

## Example

### Observation ###
[ 1.40007578e+00  9.99975709e-01 -2.84893776e-03 -6.10466062e-03
 -1.78819528e-03  3.23480177e-03 -2.65888442e-03 -9.16167907e-03
 -4.35553656e-03  7.53823327e-03 -9.76245361e-04  2.54574123e-03
 -5.49951055e-03 -2.64626504e-03 -6.26597565e-03 -5.28061974e-03
 -3.06357296e-03 -9.47762645e-03 -2.63258332e-03 -5.98598697e-03
 -4.95764945e-03 -2.70515367e-03 -2.45498772e-03 -7.51694582e-03
 -2.01429512e-03  5.43315795e-03 -1.57382131e-03 -8.80381073e-03
 -6.30824173e-03  8.62207990e-03  8.19834281e-03 -3.06493273e-03
  9.00205233e-03 -5.53816139e-03  5.55993575e-03 -8.11609328e-03
 -8.26876318e-03  6.66946909e-03  9.34501359e-03  7.80125264e-03
 -9.42482786e-03  4.91964137e-04  3.81568011e-03 -4.10158742e-03
  5.21596006e-03  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  2.30137077e+00
  2.28743068e+00  4.69114006e-02  1.08742416e-03  1.06438018e-01
 -2.35674955e-02 -2.21631442e-01  4.69218145e-02  4.35365043e+00
  8.90746237e+00  9.50485947e-02  9.07589312e-02  1.17291442e-02
  2.30943541e-04  1.25414966e-02 -1.54891967e-03 -6.46333576e-02
  8.16275070e-03  4.38909075e-01  2.26194671e+00  5.77594168e-02
  4.34214453e-02  6.60039975e-02  5.10327024e-04  8.60940426e-03
  8.50486348e-05 -3.00759274e-01  1.15872443e-02  1.89386924e-01
  6.61619413e+00  2.72712306e-01  2.29945937e-01  5.50979288e-02
 -9.51805573e-03 -1.59818937e-02 -8.21207923e-02 -9.78474033e-02
 -4.61446990e-01 -8.52615285e-01  4.75175093e+00  9.31276969e-01
  9.04353365e-01  3.08296289e-02 -3.01937677e-03 -1.63769424e-02
 -1.54721959e-01 -3.05016954e-02 -2.74135063e-01 -1.54963381e+00
  2.75569617e+00  1.04954930e+00  1.03074265e+00  2.29014044e-02
 -1.05804612e-03 -7.80527350e-03 -1.39175598e-01 -1.02400249e-02
 -1.82589578e-01 -1.34697492e+00  1.76714587e+00  2.75225223e-01
  2.34519519e-01  5.26713576e-02  8.84918567e-03 -1.49700838e-02
  7.92988856e-02 -9.27996295e-02  4.49252935e-01 -8.66078344e-01
  4.75175093e+00  9.32829639e-01  9.12723118e-01  2.38028179e-02
  2.20070352e-03 -1.37238242e-02  1.33558879e-01 -2.55279956e-02
  2.36850288e-01 -1.55714010e+00  2.75569617e+00  1.05063486e+00
  1.03818071e+00  1.65016625e-02  6.67917808e-04 -6.07770853e-03
  1.13651359e-01 -7.94475171e-03  1.48564516e-01 -1.35186069e+00
  1.76714587e+00  4.27924951e-01  3.32302323e-01  1.17103562e-01
  2.79808095e-02 -3.60534428e-02  1.73503600e-01  9.35834371e-02
 -4.08893925e-01  7.24719274e-01  1.66108048e+00  3.25281259e-01
  3.42373349e-01  1.66888640e-01  7.39511742e-02 -1.50807587e-01
  1.27114409e-01  3.27545375e-01 -2.93971245e-01  5.49691407e-01
  1.22954019e+00  4.29809010e-01  3.29179750e-01  1.23981581e-01
 -3.00837245e-02 -3.80676095e-02 -1.77943629e-01  9.92989659e-02
  4.21555696e-01  7.19919770e-01  1.66108048e+00  3.26041676e-01
  3.42963935e-01  1.70774931e-01 -7.61204500e-02 -1.52094603e-01
 -1.28457038e-01  3.31457795e-01  2.98141555e-01  5.48026262e-01
  1.22954019e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  5.53443128e-03
 -1.64338135e-03 -8.72768986e-03 -1.75283515e-03 -5.19486828e-03
 -2.00627968e-03  5.63986279e-03  6.94238697e-03 -1.50843904e-02
 -4.00209671e-03 -5.35698772e-03 -2.26255462e-03  1.38360633e-02
  6.94037275e-03 -1.48969945e-02 -4.00123035e-03 -4.29411157e-03
 -2.28902240e-03  1.06041849e-02  1.57744108e-03 -5.86150839e-03
 -4.94477204e-03 -4.04134144e-03 -2.47648581e-03  1.06422738e-02
 -3.98131644e-03 -5.75357545e-03 -7.13657381e-03 -4.05461008e-03
 -2.38637456e-03  1.06422738e-02 -3.98131644e-03 -5.75357545e-03
 -7.13657381e-03 -4.05461008e-03 -2.38637456e-03  2.17464839e-02
  1.36852399e-02 -6.50826627e-03 -3.06921700e-03 -4.18669184e-03
 -3.25426398e-03  2.17669042e-02  4.34066591e-03 -6.41995684e-03
 -6.75369662e-03 -4.19358431e-03 -3.13161523e-03  2.17669042e-02
  4.34066591e-03 -6.41995684e-03 -6.75369662e-03 -4.19358431e-03
 -3.13161523e-03  1.19984357e-02  8.14628853e-03 -1.22012951e-02
 -6.22371994e-03 -1.94610370e-03 -1.17011283e-03  1.19920584e-02
  7.80129472e-03 -1.18506281e-02 -6.22925887e-03 -2.00290733e-03
 -1.22609821e-03  8.63554934e-03 -6.13398901e-03 -9.99257868e-03
  3.29729126e-04 -3.63135617e-03 -2.45125301e-03  8.64891261e-03
 -9.86103393e-03 -1.36415781e-02  2.89867713e-04 -3.03557138e-03
 -3.05992653e-03  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]

### Action ###
[-0.30361626 -0.08565623  0.33063152 -0.20910358 -0.33288172  0.35288793
 -0.36156896  0.19782278  0.07840677  0.07975624 -0.1523676  -0.33402625
  0.24983802 -0.04836554 -0.2984188   0.16104195 -0.06633887]

### Next Observation ###
[ 1.39848913e+00  9.99950151e-01 -5.14919001e-03  8.47195113e-03
  1.18651860e-03  6.42031722e-03 -8.17172260e-02  1.53813104e-02
 -3.13419889e-02 -6.30335807e-02  4.93063337e-02 -3.73536132e-02
  1.81752919e-02  1.68350326e-02  3.83331552e-02 -3.59006147e-02
 -2.76772545e-02  1.05251246e-02 -9.69458024e-03  6.89841203e-03
 -3.55859975e-02 -1.44329439e-02 -4.11415053e-01 -1.72914658e-02
 -2.34591273e-01 -5.42614119e-01  3.07735483e+00  7.10449168e-01
  5.39886411e-02 -8.57065236e+00  2.81657391e+00 -3.50148621e+00
 -6.37636344e+00  5.00749392e+00 -5.96594355e+00  2.81657775e+00
  1.68480013e+00  4.52235385e+00 -4.39505260e+00 -2.87356488e+00
  2.23129128e+00 -8.97314425e-01  1.29108724e+00 -3.33439043e+00
 -1.36965388e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  2.29784605e+00
  2.28526335e+00  4.82879834e-02  1.27297986e-03  1.13228636e-01
 -2.41259153e-02 -2.43566164e-01  4.70013113e-02  4.35026904e+00
  8.90746237e+00  9.50420170e-02  9.16138093e-02  1.25531545e-02
  2.31133649e-04  1.51339040e-02 -9.33821111e-04 -7.79910493e-02
  5.13285108e-03  4.38939252e-01  2.26194671e+00  5.75280378e-02
  4.18060902e-02  6.46396840e-02  8.60277042e-04  7.98608500e-03
 -5.20757905e-04 -2.85212438e-01  1.34732259e-02  1.85223357e-01
  6.61619413e+00  2.74717158e-01  2.31641760e-01  5.35150344e-02
 -6.86514426e-03 -1.11768801e-02 -8.24836114e-02 -7.10842023e-02
 -4.58176821e-01 -8.59949347e-01  4.75175093e+00  9.36376593e-01
  9.09300541e-01  3.10641612e-02 -3.29630114e-03 -1.89736329e-02
 -1.55742325e-01 -3.29363072e-02 -2.75096680e-01 -1.55397826e+00
  2.75569617e+00  1.05416364e+00  1.03543630e+00  2.34645250e-02
 -2.69658206e-03 -1.98085797e-02 -1.40362963e-01 -2.59324426e-02
 -1.83756460e-01 -1.34984005e+00  1.76714587e+00  2.71679252e-01
  2.29519459e-01  5.28086567e-02  7.09387696e-03 -1.09069257e-02
  7.87382982e-02 -7.33596295e-02  4.53367239e-01 -8.54183868e-01
  4.75175093e+00  9.24990602e-01  9.04869241e-01  2.36981436e-02
  1.98138407e-03 -1.30283171e-02  1.32742502e-01 -2.30849658e-02
  2.36580320e-01 -1.55021456e+00  2.75569617e+00  1.04382203e+00
  1.03163306e+00  1.64302200e-02  1.27816330e-03 -1.16721186e-02
  1.12519759e-01 -1.53069907e-02  1.47560094e-01 -1.34751085e+00
  1.76714587e+00  4.31497595e-01  3.34950861e-01  1.18042067e-01
  2.80638660e-02 -3.61262063e-02  1.75040808e-01  9.31674605e-02
 -4.10499814e-01  7.27960345e-01  1.66108048e+00  3.24856577e-01
  3.42777960e-01  1.67198325e-01  7.37473653e-02 -1.51156054e-01
  1.26692434e-01  3.28537171e-01 -2.92882868e-01  5.49842440e-01
  1.22954019e+00  4.28148517e-01  3.27442232e-01  1.23393640e-01
 -2.91822980e-02 -3.64323394e-02 -1.77403307e-01  9.57850518e-02
  4.21142149e-01  7.18336958e-01  1.66108048e+00  3.22412408e-01
  3.40041817e-01  1.69703006e-01 -7.53994883e-02 -1.51065260e-01
 -1.26903252e-01  3.30984507e-01  2.96155453e-01  5.45105550e-01
  1.22954019e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00 -5.38303657e-01
  3.08655333e+00  6.88165796e-01 -1.80861333e+00 -2.44032049e-01
 -3.10309259e-01 -4.61495488e-01 -5.49127109e+00  8.31949021e-01
  4.12577829e-01 -2.18547155e-01  2.35218980e-02  2.34846652e+00
 -5.46426576e+00  1.03644122e+00  4.09425445e-01  1.50668803e-01
  1.80805146e-02 -4.03010958e-01 -6.53448441e-01 -5.70011139e+00
  1.11918073e+00  5.10610698e-02 -3.42945431e-01 -7.47881827e-02
  5.29760078e+00 -5.82829706e+00  3.47604393e+00 -7.94834396e-02
 -3.68694160e-01 -7.47881827e-02  5.29760078e+00 -5.82829706e+00
  3.47604393e+00 -7.94834396e-02 -3.68694160e-01 -3.01915116e-01
 -9.83455774e-01 -9.07383214e-01  2.54782990e-01  1.35673762e-01
  1.94367830e-01 -2.66308796e-01  3.40862887e+00 -9.59032421e-01
  1.97440710e+00  1.21275936e-01  1.55509832e-01 -2.66308796e-01
  3.40862887e+00 -9.59032421e-01  1.97440710e+00  1.21275936e-01
  1.55509832e-01 -2.90741669e+00  3.68909959e-01  1.19664355e+00
 -4.91501214e-01 -1.45293897e+00 -6.34788374e-01 -2.90556919e+00
  9.82254221e-01  5.41459066e-01 -4.71498269e-01 -1.35000135e+00
 -5.38367964e-01  5.06528748e-01  1.95559594e-01 -1.14526111e+00
 -6.50462421e-01  2.45388348e-01 -4.22033214e-01  5.15183691e-01
  1.16907430e+00 -1.80838615e-01 -6.36199282e-01  9.32525526e-02
 -2.68591141e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00 -8.56562331e+00
 -3.03616256e+01  3.30631524e+01 -2.09103584e+01 -3.32881719e+01
  1.05866379e+02 -7.23137915e+01  1.97822779e+01  7.84067735e+00
  2.39268713e+01 -3.04735214e+01 -8.35065618e+00  6.24595061e+00
 -1.20913861e+00 -7.46046975e+00  4.02604863e+00 -1.65847167e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
Reward: 4.895573327545365
Done: False

{STATE_ACTION_TO_PREDICT}

</user>

<assistant>
### Next Observation ###

</assistant>