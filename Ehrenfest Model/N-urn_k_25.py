# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:46:08 2024

@author: xtwan
"""

import numpy as np
import random
import scipy
import csv
import sys
import gc
from scipy.stats import expon
import matplotlib.pyplot as plt
from scipy.stats import norm



# Parameters
n_urn = 5               # Number of urns
N_ball = 5 * n_urn      # Total number of balls
theta_list = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) #np.array([1, 0.5, 0.7, 0.3, 0.9]) # Non-uniform diffusion rates for each urn

#################### Unbiased Gradient Estimator Functions ####################
def one_step_transition(urns, n_urn, N_ball, theta_list):
    """
    Perform one step in the n-urn model with non-uniform diffusion rates.
    
    Parameters:
    urns (np.array): Current configuration of balls in each urn (length n_urn).
    n_urn (int): Total number of urns.
    N_ball (int): Total number of balls.
    theta_list (list): List of diffusion parameters for each urn.
    
    Returns:
    np.array: Updated configuration of balls in each urn.
    """
    
    urns = urns.copy()
    # Select an urn to potentially move a ball from, weighted by number of balls
    urns_dist = urns / N_ball
    current_urn = np.random.choice(n_urn, p=urns_dist)
    #current_prob = urns_dist[current_urn]
    
    # Retrieve the diffusion rate for the selected urn
    theta_current = theta_list[current_urn]
    
    # Calculate transition probabilities to other urns and adjust by the current urn's theta
    target_urn_probs = np.ones(n_urn) / (n_urn)  # Uniform probability for other urns
    target_urn_probs[current_urn] = 0  # Exclude the current urn itself for movement
    target_urn_probs *= theta_current  # Apply the specific theta for the current urn

    # Calculate the probability of staying in the current urn
    stay_prob = 1 - np.sum(target_urn_probs)
    target_urn_probs[current_urn] = stay_prob  # Set the stay probability for the current urn

    # Select the target urn based on the adjusted probabilities
    new_urn = np.random.choice(n_urn, p=target_urn_probs)
    
    grad_p = np.zeros(n_urn)
    
    if new_urn != current_urn:
        tran_p = theta_current * 1 / n_urn
        grad_p[current_urn] = 1 / n_urn
    else:
        tran_p = 1 - theta_current * (n_urn - 1) / n_urn
        grad_p[current_urn] = -(n_urn - 1) / n_urn
    
    #tran_p = target_urn_probs[new_urn]  # Store the transition probability for analysis
    
    # Move the ball: decrease from current urn, increase in the new urn
    urns[current_urn] -= 1
    urns[new_urn] += 1

    return urns, tran_p, grad_p



def couple_state_transition(urns_1, urns_2, n_urn, N_ball, theta_list):
    urns_1 = urns_1.copy()
    urns_2 = urns_2.copy()
    # Select an urn to potentially move a ball from, weighted by number of balls
    urns_1_dist, urns_2_dist = urns_1 / N_ball, urns_2 / N_ball
    current_urn_1, current_urn_2 = maximal_coupling_sample(urns_1_dist, urns_2_dist)
    
    # Retrieve the diffusion rate for the selected urn
    theta_current_1 = theta_list[current_urn_1]
    theta_current_2 = theta_list[current_urn_2]
    
    # Calculate transition probabilities to other urns and adjust by the current urn's theta
    target_urn_probs_1 = np.ones(n_urn) / (n_urn)  # Uniform probability for other urns
    target_urn_probs_1[current_urn_1] = 0  # Exclude the current urn itself for movement
    target_urn_probs_1 *= theta_current_1  # Apply the specific theta for the current urn
    # Calculate transition probabilities to other urns and adjust by the current urn's theta
    target_urn_probs_2 = np.ones(n_urn) / (n_urn)  # Uniform probability for other urns
    target_urn_probs_2[current_urn_2] = 0  # Exclude the current urn itself for movement
    target_urn_probs_2 *= theta_current_2  # Apply the specific theta for the current urn

    # Calculate the probability of staying in the current urn
    stay_prob_1 = 1 - np.sum(target_urn_probs_1)
    stay_prob_2 = 1 - np.sum(target_urn_probs_2)
    target_urn_probs_1[current_urn_1] = stay_prob_1  # Set the stay probability for the current urn
    target_urn_probs_2[current_urn_2] = stay_prob_2
    
    
    # Select the target urn based on the adjusted probabilities
    new_urn_1, new_urn_2 = maximal_coupling_sample(target_urn_probs_1, target_urn_probs_2)
    
    
    grad_p_1 = np.zeros(n_urn)
    if new_urn_1 != current_urn_1:
        tran_p_1 = theta_current_1 * 1 / n_urn
        grad_p_1[current_urn_1] = 1 / n_urn
    else:
        tran_p_1 = 1 - theta_current_1 * (n_urn - 1) / n_urn
        grad_p_1[current_urn_1] = -(n_urn - 1) / n_urn
    
    
    grad_p_2 = np.zeros(n_urn)
    if new_urn_2 != current_urn_2:
        tran_p_2 = theta_current_2 * 1 / n_urn
        grad_p_2[current_urn_2] = 1 / n_urn
    else:
        tran_p_2 = 1 - theta_current_2 * (n_urn - 1) / n_urn
        grad_p_2[current_urn_2] = -(n_urn - 1) / n_urn
    
    
    urns_1[current_urn_1] -= 1
    urns_1[new_urn_1] += 1
    urns_2[current_urn_2] -= 1
    urns_2[new_urn_2] += 1
    
    return urns_1, tran_p_1, grad_p_1, urns_2, tran_p_2, grad_p_2
    
    

def maximal_coupling_sample(urns_p, urns_q):
    """
    Perform a maximal coupling of two urn configurations represented by probability distributions.
    
    Parameters:
    urns_p (np.array): Probability distribution for the first configuration (sum to 1).
    urns_q (np.array): Probability distribution for the second configuration (sum to 1).
    
    Returns:
    tuple: A pair of sampled urn indices (X, Y) where X is sampled from urns_p, 
           and Y is sampled from urns_q, maximally coupled to maximize the chance they are the same.
    """
    n_urn = len(urns_p)
    
    # Step 1: Sample X ~ p
    X = np.random.choice(n_urn, p=urns_p)
    
    # Sample W | X ~ Uniform([0, p(X)])
    W = np.random.uniform(0, urns_p[X])
    
    # Check if W <= q(X) to decide if we can couple
    if W <= urns_q[X]:
        # Maximal coupling achieved: output (X, X)
        return X, X
    else:
        # Step 2: Otherwise, sample Y from q and continue until W > p(Y)
        while True:
            # Sample Y* ~ q
            Y_star = np.random.choice(n_urn, p=urns_q)
            
            # Sample W* ~ Uniform([0, q(Y*)])
            W_star = np.random.uniform(0, urns_q[Y_star])
            
            # Check if W* > p(Y*) to accept (X, Y*)
            if W_star > urns_p[Y_star]:
                return X, Y_star
            





def cost_f(urns, n_urn, N_ball, alpha = 1.2):
    """
    Calculate the fill rate objective with a fixed occupancy threshold 
    independent of theta values.

    Parameters:
    urns (np.array): Current configuration of balls in each urn (length n_urn).
    N_ball (int): Total number of balls in the system.
    n_urn (int): Total number of urns.
    alpha (float): Multiplicative factor to adjust the occupancy threshold (default is 1.0).
    
    Returns:
    float: The fill rate, representing the proportion of urns meeting the threshold.
    """

    #return urns[-1]
    
    # Define a fixed occupancy threshold for all urns
    #fixed_threshold = [0.2, 0.2, 0.6] * (N_ball / n_urn)
    
    # Count the urns that meet or exceed the fixed threshold
    #filled_urns = sum(1 for balls in urns if balls >= fixed_threshold)
    
    # Calculate the fill rate
    #fill_rate = filled_urns / n_urn

    fill_rate = 1 if urns[-1] > alpha * (N_ball / n_urn) else 0
    
    return fill_rate
    





def poisson_solution_estimate(x, k_skeleton, base_point, n_urn, N_ball, theta_list):
    x = x.copy()
    y = base_point.copy()
    cost = cost_f(x, n_urn, N_ball) -  cost_f(y, n_urn, N_ball)
    #print('before coupling kernel')
    #print('x is {}'.format(x))
    #print('y is {}'.format(y))
    #print('cost is {}'.format(cost))
    count = 1
    
    while np.any(x != y):
        #print('poisson')
        #print(count)
        #print(x, y)
        x1, _, _, y1, _, _ = couple_state_transition(x, y, n_urn, N_ball, theta_list)
        x, y = x1, y1
        
        #print('x and y are {}'.format([x, y]))
        
        if count % k_skeleton == 0:
            cost += cost_f(x, n_urn, N_ball) -  cost_f(y, n_urn, N_ball)
        count += 1
    
    #print('after coupling kernel')
    #print('x is {}'.format(x))
    #print('y is {}'.format(y))
    #print("coupling time {}".format(count))
    
    #print('cost is {}'.format(cost))
    
    #sys.exit('coupling check')
        
    return cost, 2 * count





def estimate_derivative_new_efficient(state, N_iter, k_skeleton, n_urn, N_ball, theta_list):
    estimator_long_run_average = 0
    estimator_bias_correction = 0
    transition_time = 0
    count = 0
    
    x = state.copy()
    y = state.copy()
    base_point = state.copy()
    
    for i in range(k_skeleton):
        x1, _, _ = one_step_transition(x, n_urn, N_ball, theta_list)
        #print('x1 is {}'.format(x))
        x = x1
        
    transition_time += k_skeleton 
    coupled = False
    
    #print('x is {}'.format(x))
    
    for i in range(N_iter):
        #print(i)
        #print([x, y])
        #print(coupled)
        
        #if i == 5:
        #    sys.exit('check')
        
        count += 1
        likelihood_ratio_x = 0
        likelihood_ratio_y = 0
        
        if coupled == True:
            for j in range(k_skeleton):
                #print('sum state {}'.format(np.sum(x)))
                x1, tran_p, grad_p = one_step_transition(x, n_urn, N_ball, theta_list)
                likelihood_ratio_x += grad_p / tran_p
                x = x1
                
            #poisson estimation
            pois_x, transition = poisson_solution_estimate(x, k_skeleton, base_point, n_urn, N_ball, theta_list)
            #update the long run esimatior
            pois_times_likelihood_x = likelihood_ratio_x * pois_x 
            estimator_long_run_average += pois_times_likelihood_x 
            
            #print('estimator {}'.format(pois_times_likelihood_x))
            #if pois_times_likelihood_x > 100:
            #print(pois_x)
            #print(likelihood_ratio_x)
                #print('break')
            transition_time += k_skeleton + transition
        
        else:
            for j in range(k_skeleton):
                if np.any(x != y):
                    #print('before {}'.format(x, y))
                    x1, tran_p_x, grad_p_x, y1, tran_p_y, grad_p_y = couple_state_transition(x, y, n_urn, N_ball, theta_list)
                    likelihood_ratio_x += grad_p_x / tran_p_x
                    likelihood_ratio_y += grad_p_y / tran_p_y

                    
                    transition_time += 2
                    
                    #print('x, x1 is {}'.format([x, x1]))
                    #print('tran_p is {}'.format(tran_p_x))
                    #print('grad_p is {}'.format(grad_p_x))
                    
                else:
                    x1, tran_p, grad_p = one_step_transition(x, n_urn, N_ball, theta_list)
                    #print('x, x1 is {}'.format([x, x1]))
                    #print('tran_p is {}'.format(tran_p))
                    #print('grad_p is {}'.format(grad_p))

                    #if j == 3:
                    #    sys.exit('check')

                    likelihood_ratio_x += grad_p / tran_p
                    likelihood_ratio_y += grad_p / tran_p
                    y1 = x1
                    transition_time += 1
                    
                x = x1
                y = y1
            
            if i == 0:
                #poisson estimation
                pois_x, transition = poisson_solution_estimate(x, k_skeleton, base_point, n_urn, N_ball, theta_list)
                #update the long run esimatior
                pois_times_likelihood_x = likelihood_ratio_x * pois_x 
                estimator_long_run_average += pois_times_likelihood_x 
                transition_time += k_skeleton + transition
                
            else:
                #poisson estimation
                pois_x, transition_x = poisson_solution_estimate(x, k_skeleton, base_point, n_urn, N_ball, theta_list)
                #update the long run esimatior
                #print('likelihood_ratio, poix {}'.format([likelihood_ratio_x, pois_x]))
                pois_times_likelihood_x = likelihood_ratio_x * pois_x 
                estimator_long_run_average += pois_times_likelihood_x 
                
                
                #poisson estimation
                pois_y, transition_y = poisson_solution_estimate(y, k_skeleton, base_point, n_urn, N_ball, theta_list)
                #update the long run esimatior
                pois_times_likelihood_y = likelihood_ratio_y * pois_y
                
                pois_diff = pois_times_likelihood_x - pois_times_likelihood_y 
                estimator_bias_correction += (count - 1) * pois_diff 
                
                transition_time += k_skeleton + transition_x + transition_y
                
                
                
            if np.any(x != y) == False:
                coupled = True
                
    
    count += 1
    while coupled == False:
        likelihood_ratio_x = 0
        likelihood_ratio_y = 0
        for j in range(k_skeleton):
            if np.any(x != y):
                x1, tran_p_x, grad_p_x, y1, tran_p_y, grad_p_y = couple_state_transition(x, y, n_urn, N_ball, theta_list)
                likelihood_ratio_x += grad_p_x / tran_p_x
                likelihood_ratio_y += grad_p_y / tran_p_y
                
                transition_time += 2
                
            else:
                x1, tran_p, grad_p = one_step_transition(x, n_urn, N_ball, theta_list)
                likelihood_ratio_x += grad_p / tran_p
                likelihood_ratio_y += grad_p / tran_p
                transition_time += 1
                y1 = x1
                
            x = x1
            y = y1
        
        #poisson estimation
        pois_x, transition_x = poisson_solution_estimate(x, k_skeleton, base_point, n_urn, N_ball, theta_list)
        pois_times_likelihood_x = likelihood_ratio_x * pois_x 
        
        #poisson estimation
        pois_y, transition_y = poisson_solution_estimate(y, k_skeleton, base_point, n_urn, N_ball, theta_list)
        pois_times_likelihood_y = likelihood_ratio_y * pois_y
        
        pois_diff = pois_times_likelihood_x - pois_times_likelihood_y 
        estimator_bias_correction += (count - 1) * pois_diff 
        transition_time += k_skeleton + transition_x + transition_y
    
        if np.any(x != y) == False:
            coupled = True
        
        
    estimator = (estimator_long_run_average + estimator_bias_correction) / N_iter
    
    return [estimator[-1], 1, transition_time]
      

def mean_confidence_interval(data):
    confidence=0.90
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return [m, h]


def calculate_variance(data_set):
    mean = sum(data_set) / len(data_set)
    squared_diffs = [(x - mean) ** 2 for x in data_set]
    variance = sum(squared_diffs) / len(data_set)
    return variance




urns = np.ones(n_urn) * int(N_ball / n_urn)

time_old = 0
time_new = 0
estimate_old = 0
num_t_old = 0
estimate_new = 0
to = [] 
tn = []
to_pop = []
num_pop = []
tn_pop = []

eo = []
en = []
ss = []
ssn = [] 
eo_pop = []
en_pop = []
wv_pop = []
wv_sqrt_pop = []

n = 0
#for i in range(st_budget):
flag = 0
iteration = 20000000
inner_count = 0
batch_e = []
batch_num = []
batch_t = []
tau_population = []



k_skeleton = 25
N_iter = 100
batch_size = 100
state = urns


'''
##########Test Martingale###########
# Parameters
time_horizon = 20  # Define the time horizon for each trajectory
test_iter = 100000  # Number of trajectories to test
confidence_level = 0.95

# Storage for all martingale values across trajectories
martingale_values = np.zeros((test_iter, time_horizon))

for i in range(test_iter):
    print(i)
    x = urns
    martingale = 0
    for t_step in range(time_horizon):
        x1, tran_p, grad_p = one_step_transition(x, n_urn, N_ball, theta_list)
        martingale += grad_p / tran_p
        martingale_values[i, t_step] = martingale[-1]  # Store the martingale value at each time step
        
        # Update x for the next step
        x = x1

# Calculate the mean and confidence intervals at each time step across all trajectories
means = np.mean(martingale_values, axis=0)
std_errors = np.std(martingale_values, axis=0) / np.sqrt(test_iter)
z_score = norm.ppf(1 - (1 - confidence_level) / 2)
conf_intervals = z_score * std_errors

# Plotting the mean and confidence intervals over the time horizon
plt.figure()
plt.plot(range(time_horizon), means, label='Mean Martingale Value')
plt.fill_between(range(time_horizon), means - conf_intervals, means + conf_intervals, color='gray', alpha=0.3, label=f'{confidence_level*100}% Confidence Interval')
plt.xlabel('Time Step')
plt.ylabel('Martingale Mean')
plt.title('Martingale Mean and Confidence Interval Over Time')
plt.legend()
plt.show()

# Check martingale property
for t_step, (mean, ci) in enumerate(zip(means, conf_intervals)):
    print(f"Time Step {t_step + 1}: Mean = {mean:.4f}, Confidence Interval = ±{ci:.4f}")

sys.exit('Martingale testing with time horizon and confidence intervals')
##########Test Martingale###########
'''
'''
# Parameters
time_horizon = 20  # Define the time horizon for each trajectory
test_iter = 100000  # Number of trajectories to test
confidence_level = 0.95

# Initialize urn configurations for the initial state of each trajectory
initial_urns_1 = np.array([N_ball // n_urn] * n_urn)
initial_urns_2 = np.array([N_ball // n_urn] * n_urn)

# Storage for all martingale values across trajectories
martingale_values = np.zeros((test_iter, time_horizon))

# Main loop to generate trajectories and calculate martingale values
for i in range(test_iter):
    urns_1 = initial_urns_1.copy()
    urns_2 = initial_urns_2.copy()
    martingale = 0  # Initialize the martingale for this trajectory

    for t_step in range(time_horizon):
        # Perform coupled transition
        urns_1, tran_p_1, grad_p_1, urns_2, tran_p_2, grad_p_2 = couple_state_transition(urns_1, urns_2, n_urn, N_ball, theta_list)

        # Compute the martingale increment
        martingale += np.sum(grad_p_1 / tran_p_1) + np.sum(grad_p_2 / tran_p_2)

        # Store the martingale value at each time step
        martingale_values[i, t_step] = martingale

# Calculate the mean and confidence intervals at each time step across all trajectories
means = np.mean(martingale_values, axis=0)
std_errors = np.std(martingale_values, axis=0) / np.sqrt(test_iter)
z_score = norm.ppf(1 - (1 - confidence_level) / 2)
conf_intervals = z_score * std_errors

# Plotting the mean and confidence intervals over the time horizon
plt.figure(figsize=(10, 6))
plt.plot(range(time_horizon), means, label='Mean Martingale Value')
plt.fill_between(range(time_horizon), means - conf_intervals, means + conf_intervals, color='gray', alpha=0.3, label=f'{confidence_level*100}% Confidence Interval')
plt.xlabel('Time Step')
plt.ylabel('Martingale Mean')
plt.title('Martingale Mean and Confidence Interval Over Time')
plt.legend()
plt.show()

# Print out the mean and confidence intervals for each time step
for t_step, (mean, ci) in enumerate(zip(means, conf_intervals)):
    print(f"Time Step {t_step + 1}: Mean = {mean:.4f}, Confidence Interval = ±{ci:.4f}")

# Final check for martingale property: the mean should ideally be close to zero
print("Martingale testing complete with confidence intervals for verification.")

sys.exit('Martingale testing with time horizon and confidence intervals')


###########Test Martingale Couple#################
'''


for i in range(iteration):
    gc.collect()
    n = n + 1
    inner_count = inner_count + 1
    
    st_old = estimate_derivative_new_efficient(state, N_iter, k_skeleton, n_urn, N_ball, theta_list)
    
    estimate_old = estimate_old + st_old[0]
    num_t_old = num_t_old + st_old[1]
    time_old = time_old + st_old[2]
    to.append(time_old)
    eo.append(estimate_old / (n))
    
    print('n is {}'.format(n))
    print('computational cost: {}'.format(st_old[2]))
    print('number of terms is: {}'.format(st_old[1]))
    print('estimate is {}'.format(eo[-1]))
    
    eo_pop.append(st_old[0])
    num_pop.append(st_old[1])
    to_pop.append(st_old[2])
    
    batch_e.append(st_old[0])
    batch_t.append(st_old[2])
    
    
    if inner_count == batch_size:
        batch_e_var = calculate_variance(batch_e)
        batch_t_mean = np.mean(batch_t)
        
        wv_pop.append(batch_e_var * batch_t_mean)
        wv_sqrt_pop.append(np.sqrt(batch_e_var * batch_t_mean))
        
        batch_e = []
        batch_t = []
        
        inner_count = 0
    
    #wv_pop.append(st_old[0] * st_old[1])
    
    
    #wv_sqrt_pop.append(np.sqrt(st_old[0] * st_old[1]))
    

        if n % batch_size == 0:
            print('k is {}'.format(k_skeleton))
            sample_variance = calculate_variance(eo_pop)
            ss.append(sample_variance)
            
            A = scipy.stats.chi2.ppf(0.05, n-1, loc=0, scale=1)
            B = scipy.stats.chi2.ppf(0.95, n-1, loc=0, scale=1)
            upper = (n - 1) * sample_variance / A
            lower = (n - 1) * sample_variance / B
            
            ci_sample = mean_confidence_interval(eo_pop)
            aw = mean_confidence_interval(to_pop)
            nw = mean_confidence_interval(num_pop)
            wv = mean_confidence_interval(wv_pop)
            wv_sqrt = mean_confidence_interval(wv_sqrt_pop)

            # Data to be written in CSV
            data = [
                ['k', k_skeleton],
                ['Sample Variance', sample_variance],
                ['CI Upper Bound', upper],
                ['CI Lower Bound', lower],
                ['CI Half Width', (upper - lower) / 2],
                ['Average Estimate', ci_sample[0]],
                ['Half Width (Estimate)', ci_sample[1]],
                ['Average Work', aw[0]],
                ['Half Width (Work)', aw[1]],
                ['Average Work Variance', wv[0]],
                ['Half Width (Work Variance)', wv[1]]
            ]
            
            # File name with variable N
            filename = f"unbiased statistics_K_{k_skeleton}.csv"

            # Write to CSV file
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)

            # Existing print statements...
            print('half width for old estimator is {}'.format((upper - lower) / 2))
            print('sample_variance for old estimator is {}'.format(sample_variance))
            print('CI upper bound is {}'.format(upper))
            print('CI lower bound is {}'.format(lower))
            print('Average Estimate is {}'.format(ci_sample[0]))
            print('Half Width is {}'.format(ci_sample[1]))
            print('Average Work is {}'.format(aw[0]))
            print('Half Width is {}'.format(aw[1]))
            print('Average Work Variance is {}'.format(wv[0]))
            print('Half Width is {}'.format(wv[1]))























