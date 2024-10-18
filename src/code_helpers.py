import os
import re
import glob
import time
import pickle
from typing import List, Dict

def get_model_name(alias: str) -> str:
    """
    Returns the model name for a given alias.
    
    Parameters:
        alias (str): The alias of the model.
        
    Returns:
        str: The model name.
    """
    alias = alias.lower()
    # HuggingFace aliases
    if "deepseek" in alias or "deepcoder" in alias:
        return "deepseek-ai/deepseek-coder-33b-instruct"
    if "wizard" in alias:
        return "WizardLM/WizardCoder-33B-V1.1"
    if "mistral" in alias or "mixtral" in alias:
        if "7" in alias:
            return "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif "22" in alias:
            return "mistralai/Mixtral-8x22B-Instruct-v0.1"
        else:
            return "mistralai/Mixtral-8x7B-Instruct-v0.1" # Default to the smaller model
    if "llama" in alias:
        if "8" in alias:
            return "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            return "meta-llama/Meta-Llama-3-70B-Instruct" # Default to larger model as we only have that downloaded
    # OpenAI aliases
    if "gpt3" in alias or "gpt-3" in alias or "gpt3.5" in alias or "gpt-3.5" in alias:
        return "gpt-3.5-turbo"
    if "gpt4" in alias or "gpt-4" in alias:
        return "gpt-4-turbo"
    

def get_messages(prompt: str, splits: List[str] = ["system", "user", "assistant"], merge_system: bool = False) -> List[Dict[str, str]]:
    """
    Converts a prompt string into a list of messages for each split.
    
    Parameters:
        prompt (str): The prompt string.x
        splits (list[str]): A list of the splits to parse. Defaults to ["system", "user", "assistant"].
        merge_system (bool): Whether to merge the system messages into the user messages. Defaults to False.
        
    Returns:
        list[dict[str, str]]: A dictionary of the messages for each split.
    """
    
    messages = []
    for split in splits:
        start_tag = f"<{split}>"
        end_tag = f"</{split}>"

        start_idx = prompt.find(start_tag)
        end_idx = prompt.find(end_tag)
        if end_idx == -1:
            end_tag = f"<\\{split}>"
            end_idx = prompt.find(end_tag)
        
        # Skip if the split is not in the prompt (e.g. no system prompt)
        if start_idx == -1 and end_idx == -1:
            continue
        messages.append({
            "role": split,
            "content": prompt[start_idx + len(start_tag):end_idx].strip()
        })
    
    # If no splits at all, assume the whole prompt is a user message
    if len(messages) == 0:
        messages.append({
            "role": "user",
            "content": prompt
        })
        
    if merge_system:
        for i, message in enumerate(messages):
            if message["role"] == "system" and messages[i+1] and messages[i+1]["role"] == "user":
                messages[i+1]["content"] = message["content"] + " " + messages[i+1]["content"]
        messages = [message for message in messages if message["role"] != "system"]

    return messages

def parse_prompt(
        file_path: os.PathLike,
        splits: List[str] = ["system", "user"],
        subs: Dict[str, str] = None
    ):
    """
    Parses a prompt file and returns a dictionary of the messages for each split.

    Parameters:
        file_path (os.PathLike): The path to the prompt file.
        splits (list[str]): A list of the splits to parse. Defaults to ["system", "user"].
        subs (dict[str, str]): A dictionary of substitutions to make in the prompt file. In the prompt file, the keys
                                should be surrounded by curly braces, e.g. {key}. Defaults to None.

    Returns:
        list[dict[str, str]]: A dictionary of the messages for each split.
    """

    with open(file_path, "r") as f:
        prompt = f.read()

    if subs is not None:
        for key, value in subs.items():
            if key not in prompt:
                raise ValueError(f"Key {key} not found in prompt file.")
            prompt = prompt.replace(f"{{{key}}}", value)
    
    return get_messages(prompt, splits)

def get_completion(
    client, # openai.Client
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.8,
    n: int = 1,
    **kwargs
):

    start = time.perf_counter()
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        **kwargs
    )
    end = time.perf_counter()
    print(f"Time taken for inference: {end - start:.2f}s")
    if n==1:
        return response.choices[0].message.content.strip()
    else:
        return [choice.message.content.strip() for choice in response.choices]
    
def extract_code(completion: str):
    """
    Extracts the code from a completion.

    Parameters:
        completion (str): The completion to extract the code from.

    Returns:
        str: The extracted code.
    """

    # # Code is usually found between ```python and ``` because of Markdown formatting
    # start_tag = "```python"
    # end_tag = "```"

    # start_idx = completion.find(start_tag)
    # end_idx = completion.find(end_tag, start_idx + len(start_tag))

    # if start_idx == -1:
    #     #raise ValueError("Code not found in completion.")
    #     start_idx = 0
    #     start_tag = ''
        
    # if end_idx == -1:
    #     pass 
    
    # return completion[start_idx + len(start_tag):end_idx].strip()
    
    # 0. Define what can be at the beginning of a code block
    # This can be either an import statement (import or from), a class definition (class), a function definition (def), 
    # a comment (#), or the assignment of a variable (some_variable = some_value)
    code_start = r"(import|from|class|def|#|[a-zA-Z_][a-zA-Z0-9_]*\s*= *\S+)" # This part seems especially helpful for Mixtral. Let's keep an eye on this regex and see if it does something we don't want.
    
    # 1. Find all occurrences of "```Python or ``` python" and turn them into "```python"
    completion = re.sub(r"```\s*Python", "```python", completion, flags=re.IGNORECASE)
    
    # 2. Split completion by "```python"
    code_blocks = completion.split("```python")
    
    # 3. For each code block, check if there is an end tag (```) and remove everything after it
    for i, code_block in enumerate(code_blocks):
        end_idx = code_block.find("```")
        if end_idx != -1:
            code_blocks[i] = code_block[:end_idx]
        # Find the first match of the code_start regex
        match = re.search(code_start, code_blocks[i])
        # If there is a match, remove everything before it
        if match:
            code_blocks[i] = code_blocks[i][match.start():]
            
    # 4. Return the longest code block
    return max(code_blocks, key=len).strip()

def create_children_nodes(tree, parent_node, parent_name):
    #print('Parent name', parent_name)
    for c in parent_node.children.keys():
        child_name = parent_name+str(c)
        child_node = parent_node.children[c]
        #print('Child name', child_name)
        if child_node.expanded:
            tree.create_node(f'({child_node.visit_count},{child_node.value():.2f})', child_name, parent_name)
            tree = create_children_nodes(tree, child_node, child_name)
    return tree

def load_py_files(dir):
    print(dir)
    contents = []
    for path in glob.glob(os.path.join(dir, '*.py')):
        with open(path, "r") as f:
            contents.append(f.read())
    if len(contents) == 0:
        raise ValueError(f"No Python files found in {dir}")
    return contents

def save_tree(path, root, prefix, save_python_files=False, mcts_state=None):
    if not os.path.exists(path):
        os.makedirs(path)
    if prefix == '':
        with open(os.path.join(path, f'root.pkl'), 'wb') as f:
            pickle.dump(root, f)
    if mcts_state is not None:
        with open(os.path.join(path, 'mcts_state.pkl'), 'wb') as f:
            pickle.dump(mcts_state, f)
    if save_python_files:
        if root.full_code:
            with open(os.path.join(path, f'{prefix}.py'), 'w') as f:
                f.write(root.full_code)
        for k in root.children:
            child = root.children[k]
            save_tree(path, child, f'{prefix}{k}', save_python_files)

def load_tree(path):
    with open(os.path.join(path, 'root.pkl'), 'rb') as f:
        root = pickle.load(f)
    mcts_state_path = os.path.join(path, 'mcts_state.pkl')
    mcts_state = None
    if os.path.exists(mcts_state_path):
        with open(mcts_state_path, 'rb') as f:
            mcts_state = pickle.load(f)
    return root, mcts_state

def extract_relevant_traceback(error_message):
    traceback_lines = error_message.split('\n')
    traceback_lines = [line for line in traceback_lines if line] # Clean empty lines
    
    # # It's easier to identify the relevant traceback lines by looking for the line that starts with "File "<string>"
    # # as this is the first line of the traceback that is not part of our code
    # for i, line in enumerate(traceback_lines):
    #     if 'File "<string>' in line:
    #         break
        
    # if i == len(traceback_lines)-1:
    #     return error_message
    
    # # Here we can decide if it's useful to have the "File "<string>" line in the traceback, as it could be confusing. 
    # # If we decide to remove it, it would still be good to include the line at which the error occurred (which is printed on the same line)
    # return 'Traceback (most recent call last):\n'+'\n'.join(traceback_lines[i:])
    
    # TODO line number extraction is still a bit buggy, may want to use https://stackoverflow.com/questions/28836078/how-to-get-the-line-number-of-an-error-from-exec-or-execfile-in-python
    
    error = traceback_lines[-1]
    # Loop through the lines backwards and find the first line with "line" in it
    line_number = None
    for i, line in enumerate(reversed(traceback_lines)):
        if "line" in line:
            if 'gen_code_world_model.py' in line:
                # Error comes from env code, extract the line number using regex
                line_number = re.search(r"line (\d+)", line).group(1)
            else:
                # Error comes from our code (most likely because the return is incorrect), extract which function we called
                try:
                    next_line = traceback_lines[len(traceback_lines)-i]
                except IndexError:
                    break
                # Find code_env in next_line, it will be followed by the function name (e.g. code_env.step(x)), we want to extract the function name
                if re.search(r"code_env\.(.*?)\(", next_line):
                    function_error = re.search(r"code_env\.(.*?)\(", next_line).group(1)
                    return f"Error when calling the `{function_error}` function: {error}", None
            break
    
    error = f"Error on line {line_number}: {error}" if line_number else error
    return error, int(line_number) if line_number else None

def create_directory(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir, exist_ok=True)