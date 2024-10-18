import numpy as np
import copy 

def score_cwm(sample, simulator, init_state):
    """
    sample: valid action trajectory of shape (A,T) = (time horizon, action space)
    
    Executes the sample of an action plan in the selected environment starting from a given state 
    and returns the cumulative reward of those steps.
    """
    T = sample.shape[0]
    R = 0
    simulator.env.set_state(init_state)
    init_state_dict = simulator.save_state_dict()
    s = copy.deepcopy(init_state)
    for a in sample:
        s, r, terminated, truncated, info = simulator.step(a, s)
        R += r
        
        if terminated:
            break
        
    simulator.load_state_dict(init_state_dict)
    return R

# Cross-Entropy Method
def cross_entropy_method_cwm(
    simulator,
    init_state,
    T = 10,
    I = 10, 
    N = 1000, 
    K = 50, 
    debug=False

):
    """
    Assume simulator initialized as:
    simulator = CWMSimulatorCEM(code_env)
    simulator.action_space = real_env.action_space
    """
    # Initialization
    A = simulator.action_space.shape
    D = (T,)+A
    x_min = np.array(simulator.action_space.low).reshape(1,-1)
    x_max = np.array(simulator.action_space.high).reshape(1,-1)
    mu = np.tile((x_min+x_max)/2,(T,1))
    sigma = np.tile(np.array([max(abs(m), abs(M))/2 for m,M in zip(x_min[0],x_max[0])]), (T,1))
    
    if debug:
        print('Initial mean', mu, mu.shape) # (T, A)
        print('Initial sigma', sigma, sigma.shape) # (T, A)
        
    for i in range(I):
        # Sample from the current distribution
        samples = np.random.normal(mu, sigma, size=(N,) + D) # (N, T, A)
        #print('samples.shape', samples.shape)
        
        # Enforce (x_min, x_max) boundaries by clipping 
        clipped_samples = np.clip(samples, x_min, x_max) # (N, T, A)
        #print('clipped_samples.shape', clipped_samples.shape)
        
        # Compute scores for each sample
        scores = np.array([score_cwm(sample, simulator, init_state) for sample in clipped_samples])  # (N,)
        if (i == (I-1)//2) or (i ==(I-1)):
            print(f"Iteration {i+1} of {I} - Max score: {scores.max():.4f}")
        
        # Select elites
        elites_indices = np.argsort(-scores)[:K]
        elites = clipped_samples[elites_indices] # (K, T, A)
        #print('elites.shape', elites.shape)
        
        # Update mean and sigma using elites
        mu = np.mean(elites, axis=0)  # (T, A)
        sigma = np.std(elites, axis=0)  # (T, A)
        #print('mu.shape', mu.shape) 
        #print('sigma.shape', sigma.shape)
    
    top_elites_index = np.argsort(-scores)[0] # take top elite
    assert clipped_samples[top_elites_index].shape == mu.shape, "top sample has wrong shape!"
    return clipped_samples[top_elites_index]
    #return mu