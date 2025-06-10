#!/usr/bin/env python

# Code originally written by Alex Dombos 
#Modified to the current form by Cade Dembski (cdembski@nd.edu), March 2025
# Purpose: Calculate all quantities and plot all spectra concerning beta decay

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import math


class Point:
    def __init__(self,x, y):
        self.x = x
        self.y = y

class Level:
    def __init__(self, energy, beta_feeding, beta_feeding_error):
        self.energy = energy
        self.beta_feeding = float(beta_feeding)
        self.beta_feeding_error = float(beta_feeding_error)

    
    def __str__(self):
        return '(energy, beta_feeding) = ({}, {})'.format(self.energy, self.beta_feeding)
        
class Isotope:
    def __init__(self, Z, A, levels = None):
        self.Z = int(Z)
        self.A = int(A)
        if levels is not None:
            sum_of_beta_feedings = sum(level.beta_feeding for level in levels)
            #if sum_of_beta_feedings != 1.0:
            #if abs(1.0 - sum_of_beta_feedings) > 1e-6:
                #raise ValueError('Beta decay feeding intensities sum to {} instead of 1.0'.format(sum_of_beta_feedings))
            self._levels = levels
        else:
            self._levels = None
            
    @property
    def levels(self):
        return self._levels
    @levels.setter
    def levels(self, list_of_levels):
        sum_of_beta_feedings = sum(level.beta_feeding for level in list_of_levels)
        #if sum_of_beta_feedings != 1.0:
        #if abs(1.0 - sum_of_beta_feedings) > 1e-6:
            #raise ValueError('Beta decay feeding intensities sum to {} instead of 1.0'.format(sum_of_beta_feedings))
        self._levels = list_of_levels
        
    def __str__(self, verbose = False):
        output = '(Z = {}, A = {})'.format(self.Z, self.A)
        if verbose:
            output += '\n'
            output += '\n'.join(str(level) for level in self.levels)
        return output

class BetaDecay:
    def __init__(self, Z, A, ground_state_to_ground_state_Q_value, decay_type, half_life = None):

        if decay_type not in ('+', '-'):
            raise ValueError("The decay type must either be '+' or '-' but '{}' was selected".format(decay_type))
        self.decay_type = str(decay_type)
        
        self.parent = Isotope(Z, A)
        self.ground_state_to_ground_state_Q_value = float(ground_state_to_ground_state_Q_value)
        
        # Beta Plus Decay
        if self.decay_type == '+':
            self.daughter = Isotope(Z - 1, A)
            self.beta_sign_number = +1
            electron_mass_energy_equivalent = 0.5109989461 * 1e3 # NIST CODATA
            self.Q_value = self.ground_state_to_ground_state_Q_value - 2 * (electron_mass_energy_equivalent)
            
        # Beta Minus Decay
        if self.decay_type == '-':
            self.daughter = Isotope(Z + 1, A)
            self.beta_sign_number = -1
            self.Q_value = self.ground_state_to_ground_state_Q_value

        if half_life is not None:
            self.half_life = float(half_life)
        else:
            self.half_life = None
            
    def __str__(self):
        output = 'Beta{} Decay\nParent {}\nDaughter {}\nGround state to ground state Q value = {:.4f}\nBeta decay Q value = {:.4f}'.format(self.decay_type, self.parent, self.daughter,
                                                                                                                                           self.ground_state_to_ground_state_Q_value,
                                                                                                                                           self.Q_value)
        if self.half_life is None:
            return output
        else:
            return output + '\nHalf-life = {}'.format(self.half_life)
            
    @property
    def transitions(self):
        if self.parent.levels is None:
            self.parent.levels = [Level(0.0, 1.0, 0)] # The ground state always exists
        transitions = []
        for parent_level in self.parent.levels:
            for daughter_level in self.daughter.levels:
                transitions.append(Transition(self, parent_level, daughter_level))
        return transitions

class Transition:
    def __init__(self, beta_decay, initial, final):
        self.beta_decay = beta_decay
        self.initial = initial
        self.final = final
        electron_mass_energy_equivalent = 0.5109989461 * 1e3 # NIST CODATA
        maximum_kinetic_energy = (beta_decay.Q_value + initial.energy - final.energy) / electron_mass_energy_equivalent
        #print("Test:", maximum_kinetic_energy)
        kinetic_energy = np.linspace(start = 1e-6, stop = maximum_kinetic_energy,
                                     num = int(1e5), endpoint = True, retstep = True)
        self.KE = kinetic_energy[0]
        self.step_size = kinetic_energy[1]
        self.max_KE = maximum_kinetic_energy

    def __str__(self):
        return 'Initial: {}\tFinal: {}'.format(self.initial, self.final)
        
    def beta_phase_space_dnde(self):
        electron_mass_energy_equivalent = 1.0
        return ((self.KE**2.0 + 2.0 * self.KE * electron_mass_energy_equivalent)**0.5) * (self.max_KE - self.KE)**2.0 * (self.KE + electron_mass_energy_equivalent)

    def neutrino_phase_space_dnde(self):
        if self.beta_decay.decay_type == '-':
            raise ValueError('Neutrinos are not emitted in beta minus decay')
        neutrino_kinetic_energy = self.max_KE - self.KE
        self.KE = neutrino_kinetic_energy
        return self.beta_phase_space_dnde()

    def antineutrino_phase_space_dnde(self):
        if self.beta_decay.decay_type == '+':
            raise ValueError('Antineutrinos are not emitted in beta plus decay')
        antineutrino_kinetic_energy = self.max_KE - self.KE
        self.KE = antineutrino_kinetic_energy
        return self.beta_phase_space_dnde()
    
    def fermi_function_A(self):
        if self.beta_decay.decay_type == '+':
            raise ValueError('This Fermi function is not for beta plus decay')
        if self.beta_decay.daughter.Z < 16.0:
            m = 7.30 * 10**-2.0
            K = 9.40 * 10**-1.0
            A = m * self.beta_decay.daughter.Z + K
        if self.beta_decay.daughter.Z >= 16.0:
            a0 = 404.56 * 10.0**-3.0
            b0 = 73.184 * 10.0**-3.0
            A  = 1.0 + a0 * np.exp(b0 * self.beta_decay.daughter.Z)
        if self.beta_decay.daughter.Z <= 56.0:
            a = 5.5465 * 10.0**-3.0
            b = 76.929 * 10.0**-3.0
        if self.beta_decay.daughter.Z > 56.0:
            a = 1.2277 * 10.0**-3.0
            b = 101.22 * 10.0**-3.0
        B  = a * self.beta_decay.daughter.Z * np.exp(b * self.beta_decay.daughter.Z)
        return (A + (B / self.KE))**0.5
        
    def fermi_function_B(self):
        e  = self.KE + 1.0
        p  = (e**2.0 - 1.0)**0.5
        u  = (-1) * (self.beta_decay.beta_sign_number) * self.beta_decay.daughter.Z / 137.0
        s  = (1.0 - u**2.0)**0.5 - 1.0 # There is a -1.0 here unlike fermi_function_C
        y  = 2.0 * np.pi * u * e / p
        a1 = u * u * e * e + p * p / 4.0
        a2 = y / (1.0 - np.exp(-y))
        return a1**s * a2
    
    def fermi_function_C(self):
        e = self.KE + 1.0
        p = (e**2.0 - 1.0)**0.5
        u = (-1) * (self.beta_decay.beta_sign_number) * self.beta_decay.daughter.Z / 137.0
        s = (1.0 - u**2.0)**0.5 # No -1.0 like there is in fermi_function_B
        y = u * e / p
        R = (1.5 * pow(10.0, -15.0))*pow(self.beta_decay.daughter.A, 1.0 / 3.0) / (3.86159268 * pow(10.0, -13.0)) # Need R to be dimensionless so divide by (hbar/(mc)) NOTE: R and (hbar/(mc)) must have same units
        
        #aDN = 1.82 + 1.90 * self.daughter_Z + 0.01271 * pow(self.daughter_Z,2) - 0.00006 * pow(self.daughter_Z,3) # Below equation 30 of Logf Tables for Beta Decay by Gove and Martin
        #R = 0.002908 * pow(aDN,1.0/3.0) - 0.002437 * pow(aDN,-1.0/3.0) # Equation 30 of Logf Tables for Beta Decay by Gove and Martin
        #R = (0.5) * (1.0/137.0) * pow(aDN,1.0/3.0) # When using compare_tabulated_fermi_integral (the equation is from that paper)
	
# Gabriel Balk(06/29/2021): I added print statments to try and figure out what all of these variables are
#        print("######NEW LEVEL######")
#        print(e)
#        print(p)
#        print(u)
#        print(s)
#        print(y)
#        print(R)

        return 2.0 * (1.0 + s) * pow(2.0 * p * R, -2.0 * (1.0 - s)) * np.exp(np.pi * y) * pow(np.abs(gamma(s + (0 + 1j) * y)), 2.0) / pow(gamma(2.0 * s + 1.0), 2.0)

    def beta_phase_space_dnde_with_fermi_function(self, option):
        if option == 'A':
            return self.fermi_function_A() * self.beta_phase_space_dnde()
        elif option == 'B':
            return self.fermi_function_B() * self.beta_phase_space_dnde()
        elif option == 'C':
            return self.fermi_function_C() * self.beta_phase_space_dnde()
        else:
            raise NotImplementedError('Option {} is not implemented'.format(option))

    def average_beta_kinetic_energy(self, option):

        electron_mass_energy_equivalent = 0.5109989461 * 1e3 # NIST CODATA

        if self.beta_decay.decay_type == '-':
            average_A = (electron_mass_energy_equivalent) * sum(self.KE * normalize(self.beta_phase_space_dnde_with_fermi_function('A')))
            average_B = (electron_mass_energy_equivalent) * sum(self.KE * normalize(self.beta_phase_space_dnde_with_fermi_function('B')))
            average_C = (electron_mass_energy_equivalent) * sum(self.KE * normalize(self.beta_phase_space_dnde_with_fermi_function('C')))
            all_values = '{}\t{}\t{}'.format(average_A, average_B, average_C)
            if option == 'all':
                return all_values
            elif option == 'A':
                return average_A
            elif option == 'B':
                return average_B
            elif option == 'C':
                return average_C
            else:
                raise NotImplementedError('Option {} is not implemented'.format(option))
        elif self.beta_decay.decay_type == '+':
            average_B = (electron_mass_energy_equivalent) * sum(self.KE * normalize(self.beta_phase_space_dnde_with_fermi_function('B')))
            average_C = (electron_mass_energy_equivalent) * sum(self.KE * normalize(self.beta_phase_space_dnde_with_fermi_function('C')))
            output = '{}\t{}'.format(average_B, average_C)
            return output
            
    def fermi_integral(self, option):

        # Also include here the Benzoni integral
        # Also include here Sargent's law (Univ Aarthus thesis - with the note this works best for
        # low Z and something else nuclei (see thesis)

        # Also include Solveig Hyldegaard (Univ. Aarhus thesis) implementation of the fermi integral
        # found on page 139

        # Also include the fermi integral from the nuclear engineering paper
        
#	print(self.beta_decay.decay_type)
	
        if self.beta_decay.decay_type == '-':
#            print("test line 240")
            integral_A = sum(self.beta_phase_space_dnde_with_fermi_function('A')) * self.step_size
            integral_B = sum(self.beta_phase_space_dnde_with_fermi_function('B')) * self.step_size
            integral_C = sum(self.beta_phase_space_dnde_with_fermi_function('C')) * self.step_size
            integral_sargents_rule = pow(self.max_KE, 5.) / 30.
            all_values = '{}\t{}\t{}\t{}'.format(integral_A, integral_B, integral_C, integral_sargents_rule)
            if option == 'all':
                return all_values
            elif option == 'A':
                return integral_A
            elif option == 'B':
                return integral_B
            elif option == 'C':
#                print("Test line 253")
#	        print(integral_C)
                return integral_C
            else:
                raise NotImplementedError('Option {} is not implemented'.format(option))
		
# I made this from here 		
        if self.beta_decay.decay_type == '+':
            integral_B = sum(self.beta_phase_space_dnde_with_fermi_function('B')) * self.step_size
            integral_C = sum(self.beta_phase_space_dnde_with_fermi_function('C')) * self.step_size
#            integral_sargents_rule = pow(self.max_KE, 5.) / 30.
#            all_values = '{}\t{}\t{}\t{}'.format(integral_A, integral_B, integral_C, integral_sargents_rule)
            if option == 'B':
                return integral_B
            elif option == 'C':
#               print(integral_C)
                return integral_C
            else:
                raise NotImplementedError('Option {} is not implemented'.format(option))
#to here

            
    def beta_decay_strength(self):
        if self.beta_decay.half_life is None:
            raise ValueError('The half life is necessary for this calculation, but is None')
        return self.final.beta_feeding / (self.fermi_integral('C') * self.beta_decay.half_life)

    def log_ft(self):
        return np.log10(1.0 / self.beta_decay_strength())
        
    def B_GT(self):
        K = 6143.6
        gA_over_gV = -1.270
        return K * (1.0 / gA_over_gV)**2.0 * self.beta_decay_strength()
    
def normalize(y):
    return y / sum(y)

def BGT_calc(Z, A, ground_state_to_ground_state_Q_value, decay_type, half_life, half_life_uncertainty, energies_e, beta_feeding_arr, beta_feeding_arr_error):

    # Can experiment with incorrectly assigning beta feedings (sum != 1) in the
    # initialization to make sure an exception is raised.
    # Can also set levels with incorrect beta feedings to make sure
    # an exception is raised.
    #
    #isotope = Isotope(Z = 31, A = 76, levels = [Level(100, 0.50),
    #                                            Level(200, 0.5000)])
    #print(isotope)
    #print(isotope.__str__(verbose = True))
    #isotope.levels = [Level(0, 1.0)]
    #
    #beta_decay = BetaDecay(Z = 31, A = 76, ground_state_to_ground_state_Q_value = 6916.3, decay_type = '-', half_life = 30.6)
    #print(type(beta_decay.parent.levels))
    #print(type(beta_decay.daughter.levels))
    #beta_decay.daughter.levels = [Level(energy = 563, beta_feeding = 0.50),
    #                              Level(energy = 1108, beta_feeding = 0.50)]
    #print(beta_decay)
    #
    #for i, level in enumerate(beta_decay.daughter.levels):
    #    print(i, level)
    #
    #print("=====")
    #print(beta_decay.parent)
    #print("=====")
    ##print(beta_decay.parent.__str__(verbose = True)) # parent has no levels, so this will cause an error
    #print("=====")
    #print(beta_decay.daughter)
    #print("=====")
    #print(beta_decay.daughter.__str__(verbose = True))
    #print("=====")

#Half life is in seconds, only change below here the rest above is the math, beta_feeding is nPar
    beta_decay = BetaDecay(Z, A, ground_state_to_ground_state_Q_value, decay_type, half_life)
    beta_decay.daughter.levels = []
    
    for i in range(len(energies_e)):
        beta_decay.daughter.levels.append(Level(energy = energies_e[i]*1000, beta_feeding = beta_feeding_arr[i], beta_feeding_error = beta_feeding_arr_error[i]))
            
        
    sum_of_beta_feedings = sum(level.beta_feeding for level in beta_decay.daughter.levels)
    #print("Beta Feeding Sum:", sum_of_beta_feedings)

    #print(beta_decay)
    points = []
    errors = []
    cumulative_error = []
    cumulative_bgt = 0
    cumulative_bgt_error = 0
    #print("BGT")
    for transition in beta_decay.transitions:        
        points.append(transition.B_GT())
        #print(transition.B_GT())
    #print("Cumulative Error")
    for transition in beta_decay.transitions:
        beta_feeding_error = transition.final.beta_feeding_error
        #Combination of errors for K constant, vector/axial vector coupling ratios, and half life
        consts_err2 = (1.7/6143.6)**2+2*(.0029/1.2695)**2+(half_life_uncertainty/half_life)**2
        if transition.final.beta_feeding != 0:
            bgt_error = transition.B_GT() * np.sqrt(np.power(beta_feeding_error/transition.final.beta_feeding,2) + consts_err2)
            errors.append(bgt_error)
            cumulative_bgt_error += math.pow(bgt_error,2) 
        else:
            errors.append(0)
        #print(math.pow(cumulative_bgt_error,0.5))
        cumulative_error.append(math.pow(cumulative_bgt_error,0.5))
        
        
    return np.array(points), np.array(errors)
