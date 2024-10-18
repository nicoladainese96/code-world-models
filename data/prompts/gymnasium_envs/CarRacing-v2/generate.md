<system>
You are an experienced Python developer. You will be provided with an incomplete code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to complete the code snippet by writing the missing code so that the program performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
## Description
The easiest control task to learn from pixels - a top-down
racing environment. The generated track is random every episode.

Some indicators are shown at the bottom of the window along with the
state RGB buffer. From left to right: true speed, four ABS sensors,
steering wheel position, and gyroscope.

Remember: it's a powerful rear-wheel drive car - don't press the accelerator
and turn at the same time.

## Action Space
If continuous there are 3 actions :
- 0: steering, -1 is full left, +1 is full right
- 1: gas
- 2: breaking

If discrete there are 5 actions:
- 0: do nothing
- 1: steer left
- 2: steer right
- 3: gas
- 4: brake

## Observation Space

A top-down 96x96 RGB image of the car and race track.

## Rewards
The reward is -0.1 every frame and +1000/N for every track tile visited,
where N is the total number of tiles visited in the track. For example,
if you have finished in 732 frames, your reward is
1000 - 0.1*732 = 926.8 points.

## Starting State
The car starts at rest in the center of the road.

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