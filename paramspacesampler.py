#!/usr/bin/env python

"""Estimate bistability through random sampling of the parameter space"""

import os
import sys
import numpy as np
sys.path.append(os.getcwd())
import modelanalysis as ma
import population as pp

def sample_parameters(parameters):
    """Take a random sample from a uniform distribution for all parameters"""
    # Return a dictionary that can be used to initialise a Simulation instance
    new_params = {p: v for p, v in parameters.iteritems()}
    for p in parameters.iterkeys():
        if p.startswith("k"): # Turnover numbers
            new_params[p] = np.random.uniform(1e-1, 1e3)
        elif p.startswith("bs"):
            new_params[p] = 1
        elif p == "nbE2_0":
            new_params[p] = 0 # Should always be zero
        elif p.startswith("nb"): # Number of bursts.
            new_params[p] = np.random.uniform(1e-1, 1e4)
        elif p.startswith("K") and not p.startswith("K_"): # Affinity constants
            new_params[p] = 10 ** np.random.uniform(-5, 3) # in mmol/L
        elif p == "K_cra": # Cra-promoter Affinity
            new_params[p] = np.random.uniform(0,1e4) # Max same as max copy number?
        elif p == "n_e": # Hill coefficient for Cra-fbp binding
            new_params[p] = np.random.uniform(0,6)
        elif p == "L_fbp": # MWC allosteric constant (ratio of active/inactive forms)
            new_params[p] = 10 ** np.random.uniform(-7, 7)
        elif p == "n_fbp": # MWC cooperativity constant (integer)
            new_params[p] = np.random.randint(0, 10)
    return new_params

def analyse_bistability(parameters):
    """Has bistability been found? Returns True or False"""
    eq, stable, unstable = ma.main(parameters, plot=False)
    bi = ma.check_bistability(stable, unstable)
    return bi

def main():
    job_id = os.environ.get('JOB_ID')
    task_id = os.environ.get('SGE_TASK_ID')
    # Random choice of parameter values from uniform distribution
    rparam = sample_parameters(pp.parameters)
    # Find bistability
    bi = analyse_bistability(rparam)
    # Save: parameter values, location and size of largest step, job array id (for error identification)
    results = [rparam[p] for p in sorted(rparam.keys())]
    results.extend([job_id, task_id, bi])
    with open('../Results/paramspace/sample_{}_{}.txt'.format(job_id, task_id), 'w') as f:
        f.write("\n".join([str(n) for n in results]))
    return None

if __name__ == '__main__':
    main()
