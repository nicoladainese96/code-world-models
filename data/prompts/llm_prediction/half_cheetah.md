<system>
I want you to act as the simulator for an environment. You will be given the description of the environment and then from an initial state and action you will be tasked to simulate the environment and return the next state, reward and done signal. Only output the next state, reward and done signal and nothing else.
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

## Example

### Observation ###
[-0.12478298  0.01813205 -0.06631551  0.0094198  -0.28825312  0.11725044
  0.0822682  -0.51679118  0.31725595  0.40631022 -3.1642811   4.09749367
  0.59543823  1.41215573  4.95290803  8.02530186  0.24030582]

### Action ###
[-0.38928264 -0.03723248 -0.7244739  -0.12675352  0.76135576 -0.37873206]

### Next Observation ###
[-0.09213928 -0.07270528 -0.04546528  0.08673218 -0.4042358   0.10221561
  0.48301828 -0.33793425 -0.37283968  0.55449708 -1.35183857 -1.2216767
  0.09263673 -0.72919769 -2.13828096  6.22125638  5.13870375]
Reward: -0.2524155209944471
Done: False

### Observation ###
[ -0.07585667   0.08333078  -0.14545983  -0.11769267   0.31275676
  -0.10394819  -0.12319047   0.19966565   0.32329531   0.39465112
   3.16819635   4.57529791 -10.22914439  -6.20121217   1.52519736
  -3.53713201  -7.80042783]

### Action ###
[-0.79835343  0.73180765  0.6487395  -0.4016882  -0.9711872   0.9004741 ]

### Next Observation ###
[-0.10576098  0.17923772 -0.32301858  0.11227444  0.21261191 -0.12499025
 -0.41726682  0.15294457 -0.45417884 -0.9632038   1.39432506 -6.01797779
  9.48905046  0.49380372 -1.7120309  -6.53258115  2.53776683]
Reward: -0.5814420675104006
Done: False

{STATE_ACTION_TO_PREDICT}

</user>

<assistant>
### Next Observation ###

</assistant>