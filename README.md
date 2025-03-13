This repository contains code that can be used to extract the beta intensity and Gamow Teller strength distributions measured by the Summing NaI(Tl) (SuN) detector, plotting them with an 
associated uncertainty band. It is capable of treating contributions from neutron emission components and renormalizing out background components included within the fit (for instance, 
a spectrum for the decay of the child nucleus).

The code is designed to accept the output files of the Chi^2 minimization routine commonly used by the SuN group. These files should have everything except the lines that express the normalized
parameters after minimization (typically called npar). The files should also be named numerically based on their fit number, i.e. 1.txt, 2.txt. etc., and kept in a folder with text files 
containing the excitation energy that each parameter corresponds to within the child nucleus. 

These paths, and further variables related to the specific beta-decay under study, must be provided within the config.py file. Further instructions can be found in the comments within that file. 
Once all variables have been provided, the program can be executed with the command

```
python main.py

```

This program is written in Python3. Required packages include Numpy, Matplotlib, Pandas, Seaborn and SciPy. These modules are readily available via common package managers. Examples for the necessary formatting of the chi^2 minimization output files and excitation energy level files are also provided for the case of the beta-decay of 71Fe. 

Any questions regarding this code can be sent to cdembski@nd.edu.
