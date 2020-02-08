#/usr/bin/env python

"""Run simulations on cluster"""

import os
import sys
import copy
import subprocess
import numpy as np
import cPickle as pickle
sys.path.append(os.getcwd())
import population as pp
import sensitivity as sn

KCrafbpvals = {'KCra_fbp': [0.12, 0.1, 0.08]}
Cravals = {'nbCra' : [2.5, 5, 10], 'bsCra': [40, 20, 10]}
# For competition simulations 21/3/19, and for neutral simulations
param_vals = {'nbCra': [0.1, 1.0, 10.0, 100.0], 'bsCra': [1000, 100, 10, 1], 'KCra_fbp': [0.05]*4}
reference_strain = {'KCra_fbp':0.125, 'nbCra':5.0, 'bsCra':20}
# For evolution simulations
evo_altern = {'KCra_fbp': 0.07}
evo_cravals = {'nbCra': [0.1, 1.0, 10.0, 100.0], 'bsCra': [1000, 100, 10, 1]}
# For neutral simulations, with param_vals
neutral_altern = {'neutral': 1}
# For competition simulations with protein degradation (9/10/19)
degr_params = {'nbCra': [0.1, 1.0, 10.0, 100.0], 'bsCra': [1000, 100, 10, 1], 'gamma':[0.7]*4}
degr_ref = {'nbCra':5.0, 'bsCra':20, 'gamma':0.7}
# For population simulations 10/10/19 with protein degradation use degr_params


def cluster_submission(directory, param_vals, altern, repl=20, start=1, function='evolution'):
    """Submits an array job to the cluster. param_vals must be a dictionary
    in which the keys are the parameters to be changed, and the values a list
    containing the parameter values to be explored. If more than one parameter
    value is to be explored, the lists must be the same length. altern is the
    reference strain against which mutations compete"""
    qsub_array_call = "qsub -t {}-{} -cwd -e ../output/ -o ../output/ -S /usr/bin/python fluctenv_cluster.py ".format(start, repl)
    # The number of array jobs is determined by how many parameter sets are explored
    n_array = len(param_vals.values()[0])
    for i in range(n_array):
        params = copy.deepcopy(pp.parameters)
        for p, v in param_vals.iteritems():
            params[p] = v[i]
        with open(directory+"parameters_{}.pkl".format(i), 'w') as f:
            pickle.dump( (params, altern), f)
        # creates custom parameters and saves them in a pickle
        # Each array has a single pickle. Each replicate accesses the parameters
        # and uses it to initialise the simulation
        subprocess.call(qsub_array_call+"{} {} {} ".format(directory, i, function), shell = True)

def find_population_minimum(sim, directory, n_array, taskID):
    """Simulations from the 7/2/19 to 8/2/19; Growing a single strain in an
    environment that changes once"""
    sim.fluctuating_environments(2000, 0.1, 4*24, 48, verbose = False, save_state=False)
    # Time at which population is at its minimum / time of population growth resumption
    # Time = 0 when acetate influx starts
    popmin = sn.find_minimum(sim.results["time"], sim.results["N"], 48, 96) - 48
    # Erase pop_state. Takes up too much memory
    # sim.results["pop_state"] = None
    # Save simulation instance
    sim.save_instance("flctenv_{}_{}".format(n_array, taskID))
    np.savetxt(directory+"popmin_{}_{}.txt".format(n_array,taskID), np.array([popmin]))

def competition_Cra_noise(sim, directory, n_array, taskID, altern):
    """Compete two strains with different levels of Cra noise. KCra_fbp remains
    constant. All competing strains set up in cluster_submission
    using Cravals compete with altern"""
    #sim.KCra_fbp = 0.07
    pop = sim.initialise(2000, 0.5, altern = altern)
    # Do not record pop_state. Takes up too much memory
    # Use ten switches (20 days)
    sim.fluctuating_environments(2000, 0.1, 2*48, 48, verbose=False, pop=pop, save_state=False)
    # Save start and end and the simulation instance
    np.savetxt(directory+"pop_start_{}_{}.txt".format(n_array, taskID), pop)
    np.savetxt(directory+"pop_end_{}_{}.txt".format(n_array, taskID), sim.results['pop'])
    sim.save_instance("flctenv_{}_{}".format(n_array, taskID))

def evolution(sim, directory, n_array, taskID, altern):
    """Compete two strains until one of them is extinct"""
    if altern.keys()[0] == 'neutral':
        sim.track_locus = 11
    elif altern.keys()[0] == 'KCra_fbp':
        sim.track_locus = 8
    sim.KCra_fbp = 0.05 #Compete against {'KCra_fbp': 0.07} Initial freq 0.0005
    # ... else against {'nbCra': 5, 'bsCra': 20}
    pop = sim.initialise(2000, 0.0005, altern = altern)
    S = [0, 0]
    N = []
    glc = []
    ac = []
    pop_state = []
    timepoints = []
    strain_frequencies = [[2,2,2,2]] # To start the while loop
    while len(strain_frequencies[-1]) > 1:
        # First in glucose...
        pop, n, g, a, tp, ps, sf = sim.simulate(pop, S, 20, 0, 0.1, 48, 1/60., False, True, False)
        S = [g[-1], a[-1]]
        N.extend(n)
        glc.extend(g)
        ac.extend(a)
        strain_frequencies.extend(sf)
        if timepoints:
            timepoints.extend([timepoints[-1] + t for t in tp]) # As tp gets reinitialised
        else:
            # As timepoints is initially empty
            timepoints.extend(tp)
        # ... then in acetate
        pop, n, g, a, tp, ps, sf = sim.simulate(pop, S, 0, 40, 0.1, 48, 1/60., False, True, False)
        S = [g[-1], a[-1]]
        N.extend(n)
        glc.extend(g)
        ac.extend(a)
        strain_frequencies.extend(sf)
        timepoints.extend([timepoints[-1] + t for t in tp])
    # take away the first entry: not a real measurement
    strain_frequencies.pop(0)
    sim.results["pop"] = pop
    sim.results["N"] = N
    sim.results["glc"] = glc
    sim.results["ac"] = ac
    sim.results["time"] = timepoints
    sim.results["strain_frequencies"] = strain_frequencies
    sim.save_instance("flctenv_{}_{}".format(n_array, taskID))

if __name__ == '__main__':
    # Prepare simulation
    directory = sys.argv[1]
    n_array = int(sys.argv[2])
    function = str(sys.argv[3])
    taskID = int(os.environ["SGE_TASK_ID"])
    np.random.seed((n_array,taskID))
    # Read in parameters
    with open(directory+"parameters_{}.pkl".format(n_array), 'r') as f:
        params, altern = pickle.load(f)
    sim = pp.Simulation(params, directory)

    if function == "evolution":
        evolution(sim, directory, n_array, taskID, altern)
    elif function == "competition":
        competition_Cra_noise(sim, directory, n_array, taskID, altern)
    elif function == "population":
        find_population_minimum(sim, directory, n_array, taskID)
    else:
        raise RuntimeError("Unknown function: {}".format(function))
