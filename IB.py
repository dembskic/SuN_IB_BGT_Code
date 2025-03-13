#!/usr/bin/env python

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



#---------------------------------------------------------------------------------------------------------------------------------------------------
#Purpose: remove the contribution of the child decays from the chi^2 code output and renormalize all gamma/neutron decay components pertaining
#to the parent nucleus. 
#Array: list of dataframes containing the chi^2 output.
#Child index: integer index of the parameter corresponding to the child decay component. If no child decay, set to 'none'.
#num_neutron_pars: Integer number of neutron emission components. All should follow parent gamma decay and child decay components.
#Returns: renormalized array and array of average neutron component contributions 
def renormalize(array,child_index,num_neutron_pars):

    running_neutron_sum = np.zeros(num_neutron_pars)
    

    #iterate over all simulations
    for i in range(len(array)):
        
        print("Simulation Number ", i+1)
        #Second column contains normalized parameters, fourth column contains uncertainties in those parameters
        norm_factors=array[i][2]
        norm_factors_unc=array[i][4]

        #If the child decay is included, extract it from the file 
        if child_index != 'none':
            child_contribution = norm_factors[child_index]
            child_unc = norm_factors_unc[child_index]

        #If there is no child decay included, set its contribution to zero
        else:
            child_contribution = 0
            child_unc = 0
            
        print("Child Contribution: ", child_contribution)

        #calculatie a scaling factor to rescale all parent components to sum to 1. Propagate the uncertainty in the chilld contribution through t
        #the scaling factor
        rev_child = 1-child_contribution
        rev_child_unc = child_unc
        scaling_factor=1/rev_child
        scaling_factor_unc = rev_child_unc/rev_child*scaling_factor
        norm_factors_new=scaling_factor*norm_factors
        norm_factors_new_unc = norm_factors_new*np.sqrt((norm_factors_unc/norm_factors)**2+(scaling_factor_unc/scaling_factor)**2)
        
        #neutron components should follow daughter contribution. Sum up renormalized contributions of all neutron decay components
        if child_contribution != 0:
            neutron_contribution = norm_factors_new[child_index+1:]
            running_neutron_sum += neutron_contribution
            neutron_sum = np.sum(neutron_contribution)
            print("Neutron contribution:", neutron_sum)

            #Get all parent gamma-decay compoenents without the child decay contribution. Check to ensure the parent 
            #neutron + gamma decay components are normalized properly 

            norm_factors_new = norm_factors_new[:child_index]
            norm_factors_new_unc = norm_factors_new_unc[:child_index]
            norm_check = np.sum(norm_factors_new)+neutron_sum
            if np.allclose(norm_check,1) != True:
                print("Normalization error for fit number ", i)
                print("Total component sum: ", norm_check)

            print(" ")

        #reindexing must be done slightly differently if the child decay isn't included
        if child_contribution == 0:
            neutron_index = len(array[i][2])-num_neutron_pars
            neutron_contribution = norm_factors_new[neutron_index:]
            running_neutron_sum += neutron_contribution
            neutron_sum = np.sum(neutron_contribution)
            print("Neutron contribution:", neutron_sum)

            #Get all parent gamma-decay compoenents without the child decay contribution. Check to ensure the parent 
            #neutron + gamma decay components are normalized properly 
            norm_factors_new = norm_factors_new[:neutron_index]
            norm_factors_new_unc = norm_factors_new_unc[:neutron_index]
            norm_check = np.sum(norm_factors_new)+neutron_sum
            if np.allclose(norm_check,1) != True:
                print("Normalization error for fit number ", i)
                print("Total component sum: ", norm_check)

            print(" ")


        #Reset array to the new, renormalized factors with no child contribution
        array[i][4] = norm_factors_new_unc
        array[i][2] = norm_factors_new
        array[i] = array[i].iloc[0:len(norm_factors_new)]

    #Caclulate average contribution of each neutron component
    running_neutron_sum = np.array(running_neutron_sum/len(array))

    return array, running_neutron_sum
#---------------------------------------------------------------------------------------------------------------------------------------------------









#---------------------------------------------------------------------------------------------------------------------------------------------------
#Purpose: calculate the avereaged beta intensity distribution from all chi^2 fits. 
#Arg array: lsit of dataframes of all chi^2 fits. 
#returns: arrays for the averaged beta intensity and upper and lower uncertainties.
#cumulative average beta intensity, and upper and lower uncertainty bounds.
def calc_avg_beta_intensity(array):

    num_fits = len(array)
    num_pars = len(array[0][2])


    #new array, same length as the number of parameters/energies
    averages=np.zeros(num_pars)


    #Iterate over all energies
    for i in range(len(averages)):
    
        index_average=0
        #Iterate over all chi^2 fits
        for j in range(num_fits):
            #Sum up each fit's component value
            index_average+=array[j][2][i]
        
        #average and assign
        index_average/=num_fits
        averages[i] = index_average

    #new array for uncertainties - same shape as array
    quad_uncertainties = np.zeros((num_fits, num_pars))

    #Iterate over fits
    for i in range(len(quad_uncertainties)):
        #Iterate over parameters
        for j in range(len(quad_uncertainties[0])):
            
            #Extract statistical uncertainty output by chi^2 code
            fit_error = array[i][4][j]
            #Efficiency uncertainty - equal to 10% of the measured intensity 
            efficiency = .1 * array[i][2][j]
        
            #Combine in quadrature
            quad_uncertainties[i][j] = np.sqrt(fit_error**2+efficiency**2)


    #Arrays for lower and upper bounds of beta intensity for each fit
    low_vals = np.zeros(quad_uncertainties.shape)
    high_vals = np.zeros(quad_uncertainties.shape)

    for i in range(len(low_vals)):
        
        #Subtract/add quadrature uncertainty to each component for upper and lower values
        low_vals[i] = array[i][2] - quad_uncertainties[i]
        high_vals[i] = array[i][2] + quad_uncertainties[i]

    #Take difference between high and average, average and low
    diff_high = high_vals - averages
    diff_low = averages - low_vals

    uncertainties_high = np.zeros(num_pars)
    uncertainties_low = np.zeros(num_pars)

    #Iterate over energies
    for i in range(len(uncertainties_high)):
    
        #Treat the maximum variance as the final uncertainty in the average
        uncertainties_high[i] = np.max(diff_high[:,i])
        uncertainties_low[i] = np.max(diff_low[:,i])

    #Arrays for cumulative errors
    total_error_high = np.zeros(num_pars)
    total_error_low = np.zeros(num_pars)

    #Initial cumulative error is just the uncertainty 
    total_error_high[0] = uncertainties_high[0]
    total_error_low[0] = uncertainties_low[0]

    #Iterate over all other indices
    for i in range(1,len(total_error_high)):
    
        #Cumulative quadrature sum for all uncertainties gives total error in cumulative beta intensity
        total_error_high[i] = np.sqrt(total_error_high[i-1]**2+uncertainties_high[i]**2)
        total_error_low[i] = np.sqrt(total_error_low[i-1]**2+uncertainties_low[i]**2)
    
    #Calculate cumulative average beta intensity 
    cumulative_average = np.zeros(num_pars)
    for i in range(num_pars):
        cumulative_average[i] = np.sum(averages[0:i+1])

    #Upper and lower bounds of the cumulative beta intensity for plotting 
    upper_bound = cumulative_average + total_error_high
    lower_bound = cumulative_average - total_error_low

    #Iterate over lower bound and check if it decreases between steps. If so, flatten
    for i in range(1,len(lower_bound)):
        if lower_bound[i] < lower_bound[i-1]:
            lower_bound[i] = lower_bound[i-1]

    return averages, uncertainties_high, uncertainties_low, cumulative_average, upper_bound, lower_bound

#---------------------------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------------------------------------
#Purpose: plot cumulative beta intensity and BGT distributions with errors
#Args: path to output directory, array of energies in MeV, cumulative beta intensity array, upper and lower bounds for cumulative beta intensity
#Args continued: cumulative BGT array, upper and lower bounds for cumulative BGT
#Args continued: neutron separation energy in MeV.
#arrays for energy, cumulative IB, and cumulative BGT with neutron components included, to be plotted on top. If no neutrons, pass zero.
#Return: none
def create_plots(path, energies, IB, IB_upper, IB_lower, BGT, BGT_upper, BGT_lower, Sn, tot_energies, neutron_IB, neutron_BGT):

    plt.figure(figsize=(6,4))
    plt.step(energies, BGT, color="black", where='post', label="Experiment")
    plt.fill_between(energies, BGT_upper, BGT_lower, step='post', color="green", alpha=.4, label="Uncertainty")

    if tot_energies != 0:
        plt.step(tot_energies, neutron_BGT, color="mediumblue", where="post", label="Gamma and Neutron Decays")


    plt.ylabel("Cumulative B(GT)", fontsize=20)
    plt.xlabel("Energy (MeV)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(0,9.5)
    plt.ylim(0,3.2)

    plt.legend(fontsize=14)

    plt.savefig("{}/BGT.png".format(path), dpi=800, bbox_inches="tight")

    #Repeat for IB
    plt.figure(figsize=(6,4))
    plt.step(energies, IB, color="black", where='post', label="Experiment")
    plt.fill_between(energies, IB_upper, IB_lower, step='post', color="green", alpha=.4, label="Uncertainty")

    if tot_energies != 0:
        plt.step(tot_energies, neutron_IB, color="mediumblue", where="post", label="Gamma and Neutron Decays")
        plt.vlines(4.82, 0,1.1, color="purple", linestyle="dashed", label="Sn")


    plt.ylabel("Cumulative Beta Intensity", fontsize=20)
    plt.xlabel("Energy (MeV)", fontsize=20)
    plt.xlim(0,9.5)
    plt.ylim(0,1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=14)

    plt.savefig("{}/IB.png".format(path),  dpi=800, bbox_inches="tight")

    return 

#---------------------------------------------------------------------------------------------------------------------------------------------------




