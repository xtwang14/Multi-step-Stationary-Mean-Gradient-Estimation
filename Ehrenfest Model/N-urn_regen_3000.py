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


# Parameters
n_urn = 5                # Number of urns
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
    


def estimate_wt(s, n_urn, N_ball, theta_list):
    sum_LL = 0
    sum_tau = 0
    sum_f = 0
    regen_state = s.copy()
    
    transition_time = 0
    
    while True:
        s1, tran_p, grad_p = one_step_transition(s, n_urn, N_ball, theta_list)
        grad_p = grad_p[-1]

        sum_f += cost_f(s1, n_urn, N_ball)
        sum_tau += 1

        sum_LL += grad_p / tran_p
        transition_time += 1
        s = s1
        
        if np.any(s != regen_state) == False:
            break
    
    u = sum_f
    u_prime = sum_LL * sum_f
    l = sum_tau
    l_prime = sum_LL * sum_tau
    
    #if transition_time >= 500:
    #print('regeneration time is {}'.format(transition_time))

    return u, u_prime, l, l_prime, transition_time





def Glynn_Lecuyer_estimator(s, n_urn, N_ball, theta_list):
    estimator = 0
    transition_time = 0 
    
    u_sum = 0
    u_prime_sum = 0
    l_sum = 0
    l_prime_sum = 0
    
    for i in range(N_iter):
        #print(i)
        #dw = estimate_dw(theta, s)
        #w = estimate_w(theta, s)
        #dt = estimate_dt(theta, s)
        #t = estimate_t(theta, s)
        
        u, u_prime, l, l_prime, t2 = estimate_wt(s, n_urn, N_ball, theta_list)
        
        transition_time += t2
        
        u_sum += u
        u_prime_sum += u_prime
        l_sum += l
        l_prime_sum += l_prime
        
        #print(i)
        #print(u, u_prime, l, l_prime)
    
    
    u_estimator = u_sum / N_iter
    u_prime_estimator = u_prime_sum / N_iter
    l_estimator = l_sum / N_iter
    l_prime_estimator = l_prime_sum / N_iter
    
    
    #print('u {}'.format(u_estimator))
    #print('u prime {}'.format(u_prime_estimator))
    #print('l {}'.format(l_estimator))
    #print('l prime {}'.format(l_prime_estimator))
        
    estimator += (u_prime_estimator * l_estimator - l_prime_estimator * u_estimator) / (l_estimator ** 2)    

    return [estimator, 1, transition_time]


      

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
iteration = 200000000
inner_count = 0
batch_e = []
batch_num = []
batch_t = []
tau_population = []




N_iter = 3000
batch_size = 50

for i in range(iteration):
    gc.collect()
    n = n + 1
    inner_count = inner_count + 1
    
    st_old = Glynn_Lecuyer_estimator(urns, n_urn, N_ball, theta_list)
    
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
            print('N is {}'.format(N_iter))
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
                ['N_iter', N_iter],
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
            filename = f"regen statistics_N_{N_iter}.csv"

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






















'''

# Example usage for numerical verification
beta = 1.0
current_state = np.random.choice([-1, 1], size=(10, 10))  # Example current state
spin_index = (9, 9)  # Example spin index being updated

# Compute the analytical gradient using backpropagation
analytical_gradient = compute_gradient_transition_probability(current_state, spin_index, beta, model)

# Compute the numerical gradient by perturbing the last bias
numerical_gradient = numerical_verification(current_state, spin_index, beta, model)

# Output results
print("Analytical gradient w.r.t. last bias parameter:", analytical_gradient)
print("Numerical gradient w.r.t. last bias parameter:", numerical_gradient)
'''




















