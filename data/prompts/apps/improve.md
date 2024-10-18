<system>
You are an experienced Python developer. You will be provided with an incorrect code snippet from a Python program. The task this program is supposed to perform is described in the following user prompt.
Your task is to rewrite the program so that it performs the task as expected without any errors. You will be rewarded based on the number of test cases your code passes.
</system>

<user>
{PROB_DESCRIPTION}

Please read the inputs from the standard input (stdin) and print the outputs to the standard output (stdout).
    
First, write an explanation of the difference between the ground-truth output and the program's output in the example provided.
Secondly, point out the part of the code responsible for the incorrect prediction and why its logic is erroneous.
Third, suggest a concrete, actinable fix for it. 
Finally fix the program in its entirety following the suggestion. The expected output is in the format:
    
## Error explanation
[your explanation of the error]
    
## Error location and wrong logic
[where the error comes from and why]
    
## Fix suggestion
[how to fix the error]
    
## Correct code
```python
[your code]
```
    
## Incorrect code
You are provided with the following code snippet to fix.
```python
{CODE}
```
The code additionally makes a wrong prediction about this input.
## Input
{INPUT}
    
## Ground-truth output
{OUTPUT}
    
## Code incorrect outputs
{PREDICTION}
</user>

<assistant>
## Error explanation
</assistant>