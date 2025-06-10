#!/usr/bin/env python


#Module imports 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn 
import sys
#Visualization settings, makes good figures
custom_params = {"xtick.direction": "in", "ytick.direction": "in"}
seaborn.set_theme(style="ticks", rc=custom_params)

#Import variables, functions from config, IB, BGT files
import bgt 
from config import A,Z,half_life,half_life_uncertainty,QB,decay_mode,path,energy_file,neutron_energy_file, num_sims, child, Sn
import IB



###Load Energies###########################################################################################################################


#load array of energy levels from text file
efile_string="{}/{}".format(path,energy_file)
energies_e = np.loadtxt(efile_string)
#convert to MeV if in keV
energies_e/=1000

#If no neutrons are included, set n_neutrons to 0
if neutron_energy_file == "none":
    n_neutrons = 0

#Else load neutron energies
else:    
    nfile_string="{}/{}".format(path,neutron_energy_file)
    energies_n=np.loadtxt(nfile_string)

    n_neutrons = len(energies_n)
    energies_n/=1000




###Beta Intensity and Error Calculations###########################################################################################################################


#Load each chi^2 output into an array of dataframes that can be easily accessed
#each file should have a numeric name, i.e. 1.txt, 2.txt, etc. and consist of only the 'npar' lines of the chi^2 code output
df_array=[]
for i in range(1,num_sims+1):
    df_array.append(pd.read_csv("{}/{}.txt".format(path,i),header=None,delim_whitespace=True))


#If child decay is included, its index follows the last experimental level in the chi^2 file
if child > 0:
    child_index = np.zeros(child)
    for i in range(len(child_index)):
        child_index[i]=int(len(energies_e)+i)
#If child decay is not included, set index to none
if child == 0:
    child_index='none'


n_par_tot = len(df_array[0][2])
test_tot = len(energies_e)+child+n_neutrons


if test_tot != n_par_tot:
    print("There is a mismatch in the number of parameters")
    print("Parameters in chi^2 dataframes: ", n_par_tot)
    print("Parameters from text files: ", test_tot)
    sys.exit()



df_array, running_neutron_sum = IB.renormalize(df_array, child_index, n_neutrons)



averages, uncertainties_high, uncertainties_low, cumulative_averages, upper_bound, lower_bound = IB.calc_avg_beta_intensity(df_array)



###BGT Calculations###########################################################################################################################


#calculate average of high and low beta intensity for each bin for the BGT code 
beta_error = (uncertainties_high + uncertainties_low)/2

#Calculate BGT using function from BGT file
BGT, BGT_errors = bgt.BGT_calc(Z,A, QB,decay_mode,half_life,half_life_uncertainty,energies_e,averages,beta_error)

#Calculate cumulative BGT, cumulative BGT error, and upper/lower bounds
cumulative_BGT=np.zeros(len(BGT))
error_cumulative_BGT=np.zeros(len(BGT))
error_cumulative_BGT[0] = BGT_errors[0]

for i in range(len(BGT)):
        
    cumulative_BGT[i] = np.sum(BGT[0:i+1])

for i in range(1,len(BGT)):
        
    error_cumulative_BGT[i] = np.sqrt(error_cumulative_BGT[i-1]**2+BGT_errors[i]**2)
    
upper_bound_BGT = cumulative_BGT + error_cumulative_BGT
lower_bound_BGT = cumulative_BGT - error_cumulative_BGT

#Ensure the lower bound does not increase at any point as the cumulative BGT increases
for i in range(1,len(lower_bound_BGT)):
    if lower_bound_BGT[i] < lower_bound_BGT[i-1]:
        lower_bound_BGT[i] = lower_bound_BGT[i-1]


###Add neutron components, if present###########################################################################################################################

if (n_neutrons != 0):

    #concatenate gamma, neutron energies togther
    tot_energy = np.concatenate((energies_e,energies_n),axis=0)
    tot_non_cum = np.concatenate((averages,running_neutron_sum),axis=0)
   
    #Sort concatenated arrays into ascending energy order
    tot_energy, tot_non_cum = zip(*sorted(zip(tot_energy, tot_non_cum)))


    #Calculate cumulative distribution with all components 
    tot_cumulative = np.zeros(len(tot_energy))
    for i in range(len(tot_cumulative)):
        tot_cumulative[i] = np.sum(tot_non_cum[0:i+1])

    #Calculate BGT with neutron components added. No error since we can't really estimate this well for neutrons
    BGT_tot, BGT_errors_tot = bgt.BGT_calc(Z,A,QB,decay_mode,half_life,half_life_uncertainty,
                                           tot_energy,tot_non_cum,beta_feeding_arr_error=np.zeros_like(tot_energy))
    
    #Create cumulative BGT
    tot_cumulative_BGT=np.zeros(len(BGT_tot))
    for i in range(len(BGT_tot)):
        
        tot_cumulative_BGT[i] = np.sum(BGT_tot[0:i+1])


###Create Figures###########################################################################################################################

#path for figure output
fig_path = path+'/Figures'

#Create plots 
if (n_neutrons == 0):
    IB.create_plots(path=fig_path, energies=energies_e, IB=cumulative_averages, IB_upper=upper_bound, IB_lower=lower_bound, 
                    BGT=cumulative_BGT, BGT_upper=upper_bound_BGT, BGT_lower=lower_bound_BGT,
                    Sn=0, tot_energies=0, neutron_IB=0, neutron_BGT=0)

else:
    IB.create_plots(path=fig_path, energies=energies_e, IB=cumulative_averages, IB_upper=upper_bound, IB_lower=lower_bound, 
                BGT=cumulative_BGT, BGT_upper=upper_bound_BGT, BGT_lower=lower_bound_BGT, 
                Sn=Sn, tot_energies=tot_energy, neutron_IB=tot_cumulative, neutron_BGT=tot_cumulative_BGT)




print("Finished")










