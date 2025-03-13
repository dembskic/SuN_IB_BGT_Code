#!/usr/bin/env python

#Nucleon and Proton numbers for the parent nucleus
A = 71
Z = 26

#half-life and uncertainty
half_life = .0357 #in s
half_life_uncertainty = .002

#Beta-decay Q-value 
QB = 12400
#Decay type. Either '-' or '+' 
decay_mode = '-'


#number of simulation iterations
num_sims = 10

#Path to files output by chi2 code - named numerically
path = "/Users/cadedembski/Desktop/Research/MSU/Beta_Feeding_Code/test_71Fe"

#Name of text file with energy levels that each parameter corresponds to 
energy_file = "71_exp_levels.txt"

#Boolean for if there is a parameter corresponding to the child decay or not. 0=no, 1=yes
child = 1


#Neutron Parameters 
#Boolean for if there are neutron simulations. 0=no, 1=yes
neutrons = 1

if neutrons == 1:   
    #Name of text files with excitation energies for each neutron decay component. Should be in order that they appear in the chi^2 code, 
    #following all parent gamma-decay and child decay components
    neutron_energy_file = "Fe71_neutron_energies.txt"
if neutrons == 0:
    neutron_energy_file = "none"

#Neutron separation energy. If not including neutrons, set to 0
Sn = 5.91



