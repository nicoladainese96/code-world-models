import copy
import numpy as np

from torch import Tensor

def check_transition_via_enum(code_env, transition, possible_latents):
    transition = copy.deepcopy(transition) # trying this out
    state, action, next_state, reward, done, extra_info = transition
    compatible_latents = []
    valid = False
    for latent in possible_latents:
        code_env.set_state(copy.deepcopy(state))
        # Here we should also have the predicted valid_actions in the future
        pred_next_state, pred_reward, pred_done = code_env.step([action,latent])
        if done:
            # Ignore the next state prediction, as it's not consequential
            if pred_reward==reward and pred_done==done:
                valid = True
                compatible_latents.append(latent)
        else:
            if np.all(pred_next_state==next_state) and pred_reward==reward and pred_done==done:
                valid = True
                compatible_latents.append(latent)
        
    return valid, compatible_latents

def check_code_world_model(code_env, transitions):
    # Pass from (s,a,s',r,d) of torch variables of shape (batch_size,other_dims) to
    # [(s,a,s',r,d)_0,..., (s,a,s',r,d)_N] of numpy variables without the batch size
    if isinstance(transitions[0], Tensor):
        np_transitions = [transitions[i].cpu().numpy().astype(int) for i in range(len(transitions)) if isinstance(transitions[i], Tensor)]
    else:
        np_transitions = transitions
    transitions = list(zip(*np_transitions))
    
    possible_latents = np.arange(7) # all possible player2 actions

    all_valid = []
    all_compatible_latents = []
    for transition in transitions:
        transition_valid, compatible_latents = check_transition_via_enum(
            code_env, transition, possible_latents
        )
        all_valid.append(transition_valid)
        all_compatible_latents.append(compatible_latents)
        
    valid = np.all(all_valid)
    return valid, all_compatible_latents, all_valid