This repository contains the model described in "Gene expression noise can promote
the fixation of beneficial mutations in fluctuating environments" (Schmutzer
& Wagner). All code is written for python2.7. The files and their contents are:

population.py         The main model with stochastic gene expression
modelanalysis.py      Deterministic model with methods to find equilibrium points
test_population.py    Unit tests for both the stochastic and deterministic models
paramspacesampler.py  Parameter space sampling
fluctenv_cluster.py   Simulation set-ups used to generate most of the data in the
                      main manuscript and supporting information

To simulate an environment that switches periodically between glucose and acetate,
run the following example code:

import population as pp
sim = pp.Simulation(pp.parameters)
sim.fluctuating_environments(2000, 0.1, 96, 48, verbose=True, save_state=False)

To simulate cell lineages in an environment that is unchanging (e.g. 20 mM
acetate, no glucose):

sim.mother_machine(1000, 0, 20, 48, verbose=True, write_to_file=False)

For more examples on how to run the simulation, see the functions in
fluctenv_cluster.py

Simulation output can be found in sim.results unless otherwise specified.
