<system>
I want you to act as the simulator for an environment. You will be given the description of the environment and then from an initial state and action you will be tasked to simulate the environment and return the next state, reward and done signal. Only output the next state, reward and done signal and nothing else.
</system>

<user>
## Description

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
 in the left and right direction on the cart.

## Action Space

The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
 of the fixed force the cart is pushed with.

- 0: Push cart to the left
- 1: Push cart to the right

**Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
 the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

## Observation Space

The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

| Num | Observation           | Min                 | Max               |
|-----|-----------------------|---------------------|-------------------|
| 0   | Cart Position         | -4.8                | 4.8               |
| 1   | Cart Velocity         | -Inf                | Inf               |
| 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Angular Velocity | -Inf                | Inf               |

**Note:** While the ranges above denote the possible values for observation space of each element,
    it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
-  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
   if the cart leaves the `(-2.4, 2.4)` range.
-  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
   if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

## Rewards

Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

## Starting State

All observations are assigned a uniformly random value in `(-0.05, 0.05)`

## Episode End

The episode ends if any one of the following occurs:

1. Termination: Pole Angle is greater than ±12°
2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. Truncation: Episode length is greater than 500 (200 for v0)

## Example

### Observation ###
[-0.00741877 -0.03624395  0.03551048  0.0259908 ]

### Action ###
0

### Next Observation ###
[-0.00814365 -0.23185669  0.0360303   0.3296628 ]
Reward: 1.0
Done: False

### Observation ###
[-0.02649167 -0.6034831   0.13081776  1.0635958 ]

### Action ###
1

### Next Observation ###
[-0.03856133 -0.41031244  0.15208967  0.8146665 ]
Reward: 1.0
Done: False

### Observation ###
[-0.05891065 -0.41458005  0.19140424  0.9155581 ]

### Action ###
0

### Next Observation ###
[-0.06720225 -0.61170286  0.2097154   1.2617725 ]
Reward: 1.0
Done: True

{STATE_ACTION_TO_PREDICT}

</user>

<assistant>
### Next Observation ###

</assistant>