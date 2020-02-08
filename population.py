#!/usr/bin/env python

"""Implementation of Kotte et al (2014) model, with biomass increase and growth
on both glucose and acetate"""

import os
import sys
import numpy as np
import scipy.stats as stats
import cPickle as pickle
import scipy.integrate as spi
from collections import Counter
try:
    # For on cluster
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from collections import Counter
except TypeError:
    pass


lencell = 12 # Number of items that describe a single cell
parameters = {
# Metabolism
"KE1_glc" : 0.1,    # Michaelis-Menten dissociation constant of E1 (Gi) for glucose (mmol/L)
"KE2_ac" : 7.0,     # Michaelis-Menten dissociation constant of E2 (Ai) for acetate (mmol/L)
"KE3_pep" : 0.3,    # Michaelis-Menten dissociation constant of E3 (Lg) for PEP (mmol/g)
"KE4_pep" : 0.008,  # MWC dissociation constant of E4 (An) for PEP (mmol/g) Kotte: 0.3e-3 Works well with 0.008
"KE4_fbp" : 0.1,    # MWC dissociation constant of E4 (An) for FBP (mmol/g) Kotte: 3e-6
"L_fbp" : 4e6,      # E4 (An) MWC kinetics allosteric constant (no units)
"n_fbp" : 4,        # E4 (An) MWC cooperativity constant
"kE1_cat_s" : 100., # E1 (Gi) turnover number (/s)
"kE2_cat_s" : 100,  # E2 (Ai) turnover number (/s)
"kE3_cat_s" : 200,  # E3 (Lg) turnover number (/s)
"kE4_cat_s" : 200., # E4 (An) turnover number (/s)
# Cra activity
"n_e" : 2,          # Hill coefficient for Cra (FBP binding) (no units)
"KCra_fbp" : 0.1,   # Cra Hill dissociation constant for FBP concentration (mmol/g)
"K_cra" : 25,       # Affinity of the E2 promoter for active Cra (protein number)
# Expression parameters
"nbE1" : 10,        # Number of bursts (alpha_Gi)
"bsE1" : 500,       # Burst size (beta_Gi)
"nbE2_0" : 0,       # Number of bursts (alpha_Ai,0)
"nbE2_1" : 50,      # Number of bursts (alpha_Ai,1)
"bsE2" : 400,       # Burst size (beta_Ai)
"nbE3" : 20,        # Number of bursts (alpha_Lg)
"bsE3" : 200,       # Burst size (beta_Lg)
"nbE4" : 10,        # Number of bursts (alpha_An)
"bsE4" : 200,       # Burst size (beta_An)
"nbCra" : 5,        # Number of bursts (alpha_Cra)
"bsCra" : 20,       # Burst size (beta_Cra)
"gamma": 0          # Protein degradation rate (per hour per protein)
}

class Simulation:

    def __init__(self, parameters, directory = './'):
        """Sets parameter values and directory for saving results"""
        self.V = 1e-9  # volume of the growth chamber (L)
        self.cellmass = 3e-13  # Mass of newborn cell (g)
        self.d = 0.1  # Dilution rate (/hr)
        self.track_locus = 8 # Locus at which values should be tracked
        # Metabolism
        self.KE1_glc = parameters["KE1_glc"]     # Michaelis-Menten dissociation constant of E1 for glucose (mmol/L)
        self.KE2_ac = parameters["KE2_ac"]      # Michaelis-Menten dissociation constant of E2 for acetate (mmol/L)
        self.KE3_pep = parameters["KE3_pep"]      # Michaelis-Menten dissociation constant of E3 for PEP (mmol/g)
        self.KE4_pep = parameters["KE4_pep"]  # MWC dissociation constant of FBPase for PEP (mmol/g) Kotte: 0.3e-3
        self.KE4_fbp = parameters["KE4_fbp"]   # MWC dissociation constant of FBPase for FBP (mmol/g) Kotte: 3e-6
        self.L_fbp = parameters["L_fbp"]     # MWC kinetics shape parameters (no units)
        self.n_fbp = parameters["n_fbp"] # MWC cooperativity constant
        self.kE1_cat_s = parameters["kE1_cat_s"]  # E1 turnover number (/s)
        self.kE2_cat_s = parameters["kE2_cat_s"]   # E2 turnover number (/s)
        self.kE3_cat_s = parameters["kE3_cat_s"]  # E3 turnover number (/s)
        self.kE4_cat_s = parameters["kE4_cat_s"]  # E4 turnover number (/s)
        # Cra activity
        self.n_e = parameters["n_e"]           # Hill coefficient for Cra (FBP binding) (no units)
        self.KCra_fbp = parameters["KCra_fbp"]     # Cra Hill dissociation constant for FBP concentration (mmol/g) Kotte: 1.36e-3
        self.K_cra = parameters["K_cra"]/self.cellmass       # Affinity of the E2 promoter for active Cra (/g)
        # Expression parameters
        self.nbE1 = parameters["nbE1"]
        self.bsE1 = parameters["bsE1"]
        self.nbE2_0 = parameters["nbE2_0"]
        self.nbE2_1 = parameters["nbE2_1"]
        self.bsE2 = parameters["bsE2"]
        self.nbE3 = parameters["nbE3"]
        self.bsE3 = parameters["bsE3"]
        self.nbE4 = parameters["nbE4"]
        self.bsE4 = parameters["bsE4"]
        self.nbCra = parameters["nbCra"]
        self.bsCra = parameters["bsCra"]
        # protein degradation rate
        self.gamma = parameters["gamma"] # degradation rate per hour per protein
        # Dictionary to store results in
        self.results = {"pop": None, "N": None, "glc" : None, "ac" : None,
        "time" : None, "pop_state" : None, "strain_frequencies" : []}
        # Directory to save pop_state and instance variables in
        self.results_directory = directory



    # Equations for metabolism.
    def fructosebisphosphatase(self, E4, PEP_conc, FBP_conc, biomass):
        """Consumption of FBP to produce growth.
            E4 - number of anabolism enzymes (An)
            PEP_conc - intracellular pep concentration (mmol/g)
            FBP_conc - intracellular fbp concentration (mmol/g)
            biomass - cell mass (g)
        Returns: Rate of fbp consumption (mmol/g/hr)
        """
        v_max = E4 * self.kE4_cat_s * 3600 / (biomass * 6.023e20)
        numer = v_max * FBP_conc /self.KE4_fbp * (1 + FBP_conc / self.KE4_fbp) ** (self.n_fbp-1)
        denom = (1 + FBP_conc / self.KE4_fbp) ** self.n_fbp + self.L_fbp * (1 + PEP_conc / self.KE4_pep) ** -self.n_fbp
        return numer / denom

    def glcflux(self, E1, glc_conc, biomass):
        """Uptake of glucose to produce FBP.
            E1 - number of glucose incorporation (Gi) enzymes
            glc_conc - extracellular glucose concentration (mM)
            biomass - cell mass (g)
        Returns: Rate of glucose uptake (mmol/g/hr)"""
        kE1_cat = self.kE1_cat_s * 3600 / (biomass * 6.023e20)
        J = kE1_cat * E1 * glc_conc / (glc_conc + self.KE1_glc)
        return J

    def acflux(self, E2, ac_conc, biomass):
        """Uptake of acetate to produce FBP.
            E2 - number of acetate incorporation enzymes (Ai)
            ac_conc - concentration of extracellular acetate (mM)
            biomass - cell mass (g)
        Returns: Rate of acetate influx (mmol/g/hr)"""
        kE2_cat = self.kE2_cat_s * 3600 / (biomass * 6.023e20)
        J = kE2_cat * E2 * ac_conc / (ac_conc + self.KE2_ac)
        return J

    def pepflux(self, E3, PEP_conc, biomass):
        """Conversion of PEP into FBP.
            E3 - number of lower glycolysis enzymes (Lg)
            PEP_conc - intracellular pep concentration (mmol/g)
            biomass - cell mass (g)
        Returns: Rate of pep consumption (mmol/g/h)"""
        kE3_cat = self.kE3_cat_s * 3600 / (biomass * 6.023e20)
        return kE3_cat * E3 * PEP_conc / (PEP_conc + self.KE3_pep)

    # Protein production

    def Cra_synthesis(self, nbCra, bsCra, mu, time, biomass):
        """Constitutive, stochastic production of Cra.
            nbCra - number of Cra bursts (alpha_Cra)
            bsCra - size of Cra bursts (beta_Cra)
            mu - cell growth rate (per hr)
            time - time interval (hr)
            biomass - cell mass (g)
        Returns: Number of Cra proteins produced over time interval"""
        Cra = np.sum(np.random.geometric(1-(bsCra/float(bsCra+1)), np.random.poisson(nbCra * mu * time * biomass/self.cellmass))-1)
        return Cra

    def E1_synthesis(self, mu, time, biomass):
        """Constitutive, stochastic production of E1 (Gi)
            mu - cell growth rate (per hr)
            time - time interval (hr)
            biomass - cell mass (g)
        Returns: Number of Cra proteins produced over time interval"""
        E1 = np.sum(np.random.geometric(1-(self.bsE1/float(self.bsE1+1)), np.random.poisson(self.nbE1 * mu * time * biomass/self.cellmass))-1)
        return E1

    def E2_expected_burst_number(self, cra_active_conc):
        """Returns the expected number of E2 (Ai) mRNA synthesis events
            cra_active_conc - Concentration of CraA
        Returns: Average number of E2 bursts (alpha_Ai)"""
        p_cra_bound = cra_active_conc / (cra_active_conc + self.K_cra)
        nb = self.nbE2_0 * (1 - p_cra_bound) + self.nbE2_1 * p_cra_bound
        return nb

    def E2_synthesis(self, nb, mu, time, biomass):
        """Stochastic production of E2 (Ai).
            nb - Number of E2 bursts (alpha_Ai)
            mu - cell growth rate (per hr)
            time - time interval (hr)
            biomass - cell mass (g)
        Returns: Number of E2 proteins produced over time interval """
        # The expected number of bursts per hr (nb) is a continuous variable
        E2 = np.sum(np.random.geometric(1-(self.bsE2/float(self.bsE2+1)), np.random.poisson(nb * mu * time * biomass/self.cellmass))-1)
        return E2

    def E3_synthesis(self, mu, time, biomass):
        """Constitutive, stochastic production of E3 (Lg).
            time - time interval (hr)
            biomass - cell mass (g)
        Returns: Number of E3 proteins produced over time interval """
        E3 = np.sum(np.random.geometric(1-(self.bsE3/float(self.bsE3+1)), np.random.poisson(self.nbE3 * mu * time * biomass/self.cellmass))-1)
        return E3

    def E4_synthesis(self, mu, time, biomass):
        """Constitutive, stochastic production of E4 (An).
            time - time interval (hr)
            biomass - cell mass (g)
        Returns: Number of E4 proteins produced over time interval """
        E4 = np.sum(np.random.geometric(1-(self.bsE4/float(self.bsE4+1)), np.random.poisson(self.nbE4 * mu * time * biomass/self.cellmass))-1)
        return E4

    # Simulating cell growth and nutrient uptake from environment

    def cell_growth(self, pop, time, glc_conc, ac_conc, return_all=True):
        """Simulate growth of each cell in the population, and the rate of
        glucose and acetate uptake.
            pop - population array
            time - time interval of simulation
            N - population size
            glc_conc - concentration of glucose in environment
            ac_conc - concentration of acetate in environment
            return_all - True returns all changes and internal fluxes
                         False returns only changes in population
        Returns:
            Change in population, including biomasses and intracellular
            concentrations
            Changes in extracellular concentrations
            Intracellular fluxes
            Growth rates
        """
        biomasses = pop[:,0]
        FBP = pop[:,1]
        PEP = pop[:,2]
        Cra = pop[:,3]
        E1 = pop[:,4]
        E2 = pop[:,5]
        E3 = pop[:,6]
        E4 = pop[:,7]
        # Calculate rate of change
        J_glc = self.glcflux(E1, glc_conc, biomasses) # mmol/g/hr
        J_ac = self.acflux(E2, ac_conc, biomasses) # mmol/g/hr
        J_pep = self.pepflux(E3, PEP, biomasses) # mmol/g/hr
        J_fbp = self.fructosebisphosphatase(E4, PEP, FBP, biomasses) #mmol/g/hr
        growth_rates = 0.0896 * J_fbp # /hr
        # Production and conversion - dilution due to cell growth
        dFBPdt = J_glc + 1/2. * J_pep - J_fbp - growth_rates * FBP # mmol/g/hr
        dPEPdt = 1/2. * J_ac - J_pep - growth_rates * PEP # mmol/g/hr
        dBiodt = growth_rates * biomasses # g/hr
        # Gather together
        dpopdt = np.zeros(pop.shape)
        dpopdt[:,0] = dBiodt
        dpopdt[:,1] = dFBPdt
        dpopdt[:,2] = dPEPdt
        # Sum uptakes
        glc_uptake = np.sum(J_glc * pop[:,0]) / self.V
        ac_uptake = np.sum(J_ac * pop[:,0]) / self.V
        if return_all:
            return dpopdt, glc_uptake, ac_uptake, J_glc, J_ac, J_pep, J_fbp, growth_rates
        else:
            return dpopdt

    def metabolism(self, state_var, time, N, state_const, glc_in, ac_in, dil):
        """System of ODE's of substrate concentrations, and the metabolite uptake
        and growth of each cell. Serves as wrapper for cell_growth and calculates
        changes in external metabolite concentrations.
            state_var - system state, combines glc and ac concentrations with
                        parts of population array that specify intracellular
                        concentrations
            time - time interval (hr)
            N - population size
            state_const - part of population array that contains cell properties
                          that will not change here (protein amounts etc)
            glc_in - concentration inflowing glucose (mM)
            ac_in - concentration inflowing acetate (mM)
            dil - dilution rate (per hr)
        Returns: new system state"""
        glc = state_var[0] # mmol/L
        ac = state_var[1] # mmol/L
        mstate = state_var[2:].reshape((N,3))
        pop = np.hstack( (mstate, state_const) )
        dpopdt, glc_uptake, ac_uptake, J_glc, J_ac, J_pep, J_fbp, gr = self.cell_growth(pop, time, glc, ac)
        dglcdt = glc_in * dil - glc_uptake - glc * dil
        dacdt = ac_in * dil - ac_uptake - ac * dil
        dmstatedt = dpopdt[:,:3]
        state = np.hstack( ([dglcdt, dacdt], dmstatedt.flatten()) )
        return state

    def cra_activity(self, Cra, FBP, KCra_fbp):
        """Determine Cra activity CraA.
            Cra - number of Cra proteins per cell
            FBP - intracellular fbp concentration (mmol/g)
            KCra_fbp - Cra-fbp dissociation constant (mmol/g)
        Returns: CraA"""
        Cra_active = Cra * (1 - (FBP) ** self.n_e / ( (FBP) ** self.n_e + KCra_fbp ** self.n_e))
        return Cra_active

    def protein_synthesis(self, cell, growth_rate, time, cra_stochastic=True, CraA=None):
        """Wrapper for stochastic protein production. Estimates Cra activity.
            cell - current state of the cell
            growth_rate - cell growth rate (per hr)
            time - time interval (h)
            cra_stochastic - True makes cra numbers stochastic
            CraA - fix cra activity to be independent of fbp and Cra concentrations
        Returns: New cell state (after protein synthesis)"""
        # Unpack
        biomass = cell[0]
        FBP = cell[1]
        PEP = cell[2]
        Cra = cell[3]
        E1 = cell[4]
        E2 = cell[5]
        E3 = cell[6]
        E4 = cell[7]
        KCra_fbp = cell[8]
        nbCra = cell[9]
        bsCra = cell[10]
        if cra_stochastic:
            Cra += self.Cra_synthesis(nbCra, bsCra, growth_rate, time, biomass)
        else:
            # To have a constant Cra concentration, peg to biomass
            # This makes cra copy number a continuous variable
            Cra = nbCra * bsCra * biomass/self.cellmass

        if CraA is not None:
            # To have a constant concentration of active Cra, peg to biomass
            Cra_active = CraA * biomass/self.cellmass
        else:
            Cra_active = self.cra_activity(Cra, FBP, KCra_fbp)
        e2nb = self.E2_expected_burst_number(Cra_active/biomass)
        E1 += self.E1_synthesis(growth_rate, time, biomass)
        # For E2, add gamma to synthesis rate in order to make sure that the
        # copy number remains constant
        E2 += self.E2_synthesis(e2nb, (growth_rate + self.gamma), time, biomass)
        E3 += self.E3_synthesis(growth_rate, time, biomass)
        E4 += self.E4_synthesis(growth_rate, time, biomass)
        return [biomass, FBP, PEP, Cra, E1, E2, E3, E4, KCra_fbp, nbCra, bsCra]

    def protein_degradation(self, cell, time):
        """Degradation of E2 (Ai) at a constant rate.
            cell - current state of the cell
            time - time interval (h)
        Returns: New cell state (after degradation of E2/Ai)"""
        cell[5] = np.random.binomial(cell[5], np.e ** (-self.gamma * time))
        return cell

    def cell_division(self, pop):
        """For each cell in the population: (a) Decide if cell will divide
        (b) if so, assume that offspring cells have the same metabolite concentrations
        (c) exactly half the biomass of the parent cell and (d) proteins are
        inherited through binomial partitioning.
            pop - population array
        Returns: new population array"""

        dividing = pop[pop[:,0] >= 2 * self.cellmass]
        nondividing = pop[pop[:,0] < 2 * self.cellmass]

        cells1 = np.full(dividing.shape, np.nan)
        cells2 = np.full(dividing.shape, np.nan)
        # Biomass is halved
        cells1[:,0] = 0.5 * dividing[:,0]
        # Metabolite concentrations remain constant
        cells1[:,1:3] = dividing[:,1:3]
        # Proteins are partitioned randomly
        proteins = dividing[:, 3:8].astype(int)
        cells1[:,3:8] = np.random.binomial(proteins, 0.5)
        # Siblings have to share the parent cells' content
        cells2[:,0] = dividing[:,0] - cells1[:,0]
        cells2[:,1:3] = dividing[:,1:3]
        cells2[:,3:8] = dividing[:,3:8] - cells1[:,3:8]
        # Offspring inherit their parents KCra_fbp and other parameters
        cells1[:,8:lencell] = dividing[:,8:lencell]
        cells2[:,8:lencell] = dividing[:,8:lencell]
        # Assemble population
        offspring = np.vstack( (cells1, cells2) )
        pop = np.vstack( (nondividing, offspring) )
        return pop

    def cell_division_mother_machine(self, pop):
        """Cell division in a constant environment. Only track one of the daughter
        cells.
            pop - population array
        Returns: new population array"""
        # Get index of all dividing cells and modify cells in place
        index = pop[:,0] >= 2 * self.cellmass
        # Biomass is halved
        pop[index,0] = 0.5 * pop[index,0]
        # Metabolite concentrations remain constant
        pop[index,1:3] = pop[index,1:3]
        # Proteins are partitioned randomly
        proteins = pop[index, 3:8].astype(int)
        pop[index,3:8] = np.random.binomial(proteins, 0.5)
        # Offspring inherit their parents KCra_fbp and other parameters
        pop[index,8:lencell] = pop[index,8:lencell]
        return pop

    def cell_efflux(self, pop, dilution, time):
        """Removal of cells by being flushed out of the chemostat
            pop - population array
            dilution - rate of throughflow of medium and cell removal (per hr)
            time - time interval (h)
        Returns: population array"""
        N = len(pop)
        if N == 0:
            return pop
        else:
            # Probability of survival decays exponentially
            # See also: http://hplgit.github.io/INF5620/doc/pub/sphinx-decay/._main_decay008.html#stochastic-model
            n_survivors = np.random.binomial(N, np.e ** (-dilution * time))
            index = np.random.choice(N, n_survivors, replace = False)
            pop = pop[index]
        return pop

    def chemostat(self, pop, S, glc_in, ac_in, dilution, time=1/60.):
        """Simulate growth in an environment with constant throughflow of medium.
        This set-up is inspired by chemostats.
            pop - population array, e.g. from initialise()
            S - initial substrate concentrations (mM)
            glc_in - concentration of incoming glucose (mM)
            ac_in - concentration of incoming acetate (mM)
            dilution - dilution rate (per hr)
            time - time interval (hr)
        Returns: After the simulated interval
            population array
            substrate concentrations
            growth rates
        """
        t = np.array([0, time])
        # Retrieve all biomasses. This is needed to estimate the specific growth
        # rate.
        pop = np.array(pop)
        N = pop.shape[0]
        if N > 0:
            biomasses_old = np.empty(len(pop))
            biomasses_old[:] = pop[:, 0]
        # Update metabolic state of the system. This is both the metabolite
        # concentrations in the chemostat and within cells, and the resulting
        # biomass
        # Produce a one-dimensional array [glc, ac] + [cell, cell ...]
        mstate = pop[:,:3]
        rstate = pop[:,3:]
        state = np.hstack((S, mstate.flatten()))
        state = spi.odeint(self.metabolism, state, t, args=(N, rstate, glc_in, ac_in, dilution))
        S = state[-1, :2]
        pop[:,:3] = state[-1,2:].reshape( (N, 3) )
        # Check if any component is negative. If this is the case, return an error
        if (S < 0).any() or (pop[:,:-1] < 0).any():
            raise RuntimeError('State has negative concentrations')
        if (S < 1 / (self.V * 6.022e20)).any(): # less than one molecule per growth chamber mmol/L
            S[S < 1 / (self.V * 6.022e20)] = 0
        if (pop[:,1:lencell-1] < 1 / ((pop[:,0] * 6.022e20))[:,None]).any(): # less than one molecule per cell mmol/g
            pop[:,1:lencell-1][pop[:,1:lencell-1] < (1 / (pop[:,0] * 6.022e20))[:,None]] = 0
        # Update protein levels
        # Estimate the specific growth rate from the change in biomass (g/g/hr)
        if N > 0:
            growth_rates = (pop[:,0] - biomasses_old) / (biomasses_old * time)
            for i in range(N):
                pop[i,:lencell-1] = self.protein_synthesis(pop[i], growth_rates[i], time)
                pop[i] = self.protein_degradation(pop[i], time)
            # Randomly remove cells from population
            pop = self.cell_efflux(pop, dilution, time)
            # Reproduction of survivors
            pop = self.cell_division(pop)
        return pop, S, growth_rates

    def initialise(self, N_init, p = 1, altern = {}):
        """Initialises population of size N_init. p sets the proportion of cells
        carrying an alternative value for KCra_fbp, nbCra and bsCra set by altern.
            N_init - population size
            p - fraction of alternative cells
            altern - parameters that should be different in subpopulation of size
                     p and the alternative values of these parameters
        Returns: population array"""
        pop = np.zeros( (N_init, lencell) )
        # Start with cell specific parameter values - this info will be used to
        # determine the initial Cra copy number
        pop[:,8] = np.repeat(self.KCra_fbp, N_init)
        pop[:,9] = np.repeat(self.nbCra, N_init)
        pop[:,10] = np.repeat(self.bsCra, N_init)
        pop[:,11] = np.repeat(0, N_init)
        if altern.has_key("KCra_fbp"):
            pop[:int(p*N_init),8] = np.repeat(altern["KCra_fbp"], int(p*N_init))
        if altern.has_key("nbCra"):
            pop[:int(p*N_init),9] = np.repeat(altern["nbCra"], int(p*N_init))
        if altern.has_key("bsCra"):
            pop[:int(p*N_init),10] = np.repeat(altern["bsCra"], int(p*N_init))
        if altern.has_key("neutral"):
            pop[:int(p*N_init),11] = np.repeat(altern["neutral"], int(p*N_init))
        pop[:,0] = np.random.uniform(1,2, N_init) * self.cellmass
        pop[:,1] = np.random.uniform(0.1, 2, N_init) # Some FBP
        pop[:,3] = np.random.negative_binomial(self.nbCra, 1-self.bsCra/float(self.bsCra+1), N_init)
        if altern.has_key("nbCra") or altern.has_key("bsCra"):
            # These cells start with a different Cra distribution
            pop[:int(p*N_init),3] = np.random.negative_binomial(altern["nbCra"], 1-altern["bsCra"]/float(altern["bsCra"]+1), int(p*N_init))
        pop[:,4] = np.random.negative_binomial(self.nbE1, 1-self.bsE1/float(self.bsE1+1), N_init)
        pop[:,5] = np.random.negative_binomial(np.maximum(np.repeat(1,N_init),
            self.E2_expected_burst_number(self.cra_activity(pop[:,3], pop[:,1], pop[:,8])/pop[:,0])),
            1-self.bsE2/float(self.bsE2+1), N_init)
        pop[:,6] = np.random.negative_binomial(self.nbE3, 1-self.bsE3/float(self.bsE3+1), N_init)
        pop[:,7] = np.random.negative_binomial(self.nbE4, 1-self.bsE4/float(self.bsE4+1), N_init)
        return pop

    def simulate(self, pop, S_init, glc_in, ac_in, dilution, time, timestep=1/60., verbose=False, Return=False, save_state=True, return_growth_rates=False):
        """Simulates growth in a chemostat with an environment defined by the
        influx, efflux and cell uptake of glucose and acetate.
            pop - cell population, e.g. output of initialise()
            S_init - initial concentrations of glucose and acetate in a list
            glc_in - concentration incoming glucose
            ac_in - concentration incoming acetate
            dilution - dilution rate of chemostat
            time - duration of simulated time
            timestep - interval between simulation steps
            verbose - True print current simulation time to console
            Return - True returns observations, False stores them in self.results
            save_state - True stores population state at every timestep. Very memory intensive
            return_growth_rates - True returns growth rate of every cell at each timestep
        Returns:
            population array
            population size over time
            glucose concentrations over time
            acetate concentrations over time
            timepoints
            population state at each timepoint
            size of subpopulations/strains
            growth rate at each timepoint
        """
        S = S_init
        glucose = []
        acetate = []
        N = []
        growth_rates = []
        pop_state = []
        timepoints = []
        strain_frequencies = []

        # Estimate number of time steps
        timesteps = int( np.ceil(time / float(timestep) ) )
        for t in range(timesteps):
            if verbose:
                sys.stdout.write("\r At time {:.0f}:{:02.0f}hr of {:.0f}hrs    ".format(t*timestep//1., t*timestep%1.*60, time))
                sys.stdout.flush()
            N.append(pop.shape[0])
            glucose.append(S[0])
            acetate.append(S[1])
            if save_state:
                pop_state.append(pop)
            timepoints.append( (t+1)*timestep)
            strain_frequencies.append(Counter(pop[:,self.track_locus]))
            # NB: This advances pop and S, but no record is made. Makes sense only if
            # called from fluctuating_environments
            pop, S, gr = self.chemostat(pop, S, glc_in, ac_in, dilution, timestep)
            growth_rates.append(gr)
        if verbose:
            sys.stdout.write("\r At time {:.0f}:{:02.0f}hr of {:.0f}hrs".format(timesteps//(1/timestep), timesteps%(1/timestep), time))
            sys.stdout.flush()
        if Return == True and return_growth_rates == False:
            return pop, N, glucose, acetate, timepoints, pop_state, strain_frequencies
        elif Return == True and return_growth_rates == True:
            return pop, N, glucose, acetate, timepoints, pop_state, strain_frequencies, growth_rates
        else:
            self.results["pop"] = pop
            self.results["N"] = N
            self.results["glc"] = glucose
            self.results["ac"] = acetate
            self.results["time"] = timepoints
            self.results["pop_state"] = pop_state
            self.results['growth_rates'] = growth_rates
            self.results["strain_frequencies"] = strain_frequencies

    def wrapper_mother_machine(self, time, mstate, N, rstate, glc_conc, ac_conc):
        """Wrapper for mother_machine ode implementation, in order to keep cell_growth
        compatible with odeint.
            time - time interval (h)
            mstate - part of population array containing intracellular metabolite
                     concentrations and cell mass
            rstate - part of population array that will not change here (eg
                     protein amounts)
            glc_conc - concentration of extracellular glucose (mM)
            ac_conc - concentration of extracellular acetate (mM)
        Returns: Change in intracellular metabolite concentrations"""
        mstate = mstate.reshape((N,3))
        pop = np.hstack( (mstate, rstate) )
        dpopdt = self.cell_growth(pop, time, glc_conc, ac_conc, False)
        return dpopdt[:,:3].flatten()

    def mother_machine(self, N, glc_in, ac_in, time, timestep=1/60., verbose=True,
            save_state=False, write_to_file=False, cra_stochastic=True, CraA=None):
        """Track the growth of single cell lineages in a constant environment,
        as in a mother-machine.
            N - number of cell lineages to track
            glc_in - concentration incoming glucose (mM)
            ac_in - concentration incoming acetate (mM)
            time - duration of simulated time (h)
            timestep - interval between simulation steps (h)
            verbose - True print current simulation time to console
            save_state - record cell state at every time point. Very memory intensive
            write_to_file - save growth rates in file at every time point
            cra_stochastic - True makes cra numbers stochastic
            CraA - fix cra activity to be independent of fbp and Cra concentrations
        Returns: Observations stored in self.results"""
        # track pop_state and growth rates
        if N > 0:
            pop = self.initialise(N)
        elif self.results['pop'] is not None:
            pop = self.results['pop']
            N = len(pop)
        else:
            raise IOError("No population available. Set N > 0")
        if write_to_file: # create file
            with open(self.results_directory+"mother_machine_run.txt", 'w') as f:
                f.write("")
        timepoints=[0]
        pop_state=[np.empty(pop.shape)]
        pop_state[0][:] = pop[:]
        growth_rates=[self.cell_growth(pop, time, glc_in, ac_in, True)[7]]
        timesteps = int( np.ceil(time/float(timestep)))
        # Create integrator
        num = spi.ode(self.wrapper_mother_machine).set_integrator("vode", method="bdf", atol=1e-20)
        for t in range(1,timesteps+1):
            if verbose:
                sys.stdout.write("\r At time {:.0f}:{:02.0f}hr of {:.0f}hrs    ".format(t*timestep//1., t*timestep%1.*60, time))
                sys.stdout.flush()
            # Update metabolism
            # Produce a one-dimensional array [glc, ac] + [cell, cell ...]
            mstate = pop[:,:3]
            rstate = pop[:,3:]
            num.set_initial_value(mstate.flatten(), 0).set_f_params(N, rstate, glc_in, ac_in,)
            ode_out = num.integrate(num.t+timestep)
            pop[:,:3] = ode_out.reshape( (N, 3) )
            # Update protein copy number
            # Estimate the specific growth rate from the change in biomass (g/g/hr)
            grs = (pop[:,0] - pop_state[-1][:,0]) / (pop_state[-1][:,0] * timestep)
            for i in range(N):
                pop[i,:-1] = self.protein_synthesis(pop[i], grs[i], timestep, cra_stochastic=cra_stochastic, CraA=CraA)
                pop[i] = self.protein_degradation(pop[i], timestep)
            # Cell division
            pop = self.cell_division_mother_machine(pop)
            # Track changes in cell characteristics
            if write_to_file: # add to file
                pop_state=[np.empty(pop.shape)]
                pop_state[0][:] = pop[:]
                with open(self.results_directory+"mother_machine_run.txt", 'a') as f:
                    f.write("{},".format(t*timestep)+",".join([str(g) for g in grs])+"\n")
            else:
                timepoints.append(t * timestep)
                if save_state:
                    popcopy = np.empty(pop.shape)
                    popcopy[:] = pop[:]
                    pop_state.append(popcopy)
                else:
                    pop_state=[np.empty(pop.shape)]
                    pop_state[0][:] = pop[:]
                growth_rates.append(grs)
        # store results at end of simulation
        if not write_to_file:
            self.results['time'] = np.array(timepoints)
            self.results['growth_rates'] = np.array(growth_rates)
            if save_state:
                self.results['pop_state'] = pop_state
        self.results['pop'] = pop

    def fluctuating_environments(self, N_init = 100, dil=0.1, time=24, switch=6, timestep=1/60., verbose=False, pop=np.array([]), save_state=False, save_growth=False):
        """Simulate growth in an fluctuating environment which switches in carbon
        source periodically.
            N_init - Initial population size
            dil - dilution rate of the medium (per h)
            time - duration of the simulation (h)
            switch - time between switches in carbon source (h)
            timestep - interval between simulation steps (h)
            verbose - print current time and carbon source
            pop - initial population array. If given, N_init is ignored
            save_state - If True, store population state at each timestep. Very memory intensive
            save_growth - If True, store growth rates for each cell every timestep
        Returns: Observations stored in self.results
        """
        # The switching times only give the right endpoint for integer hours
        if len(pop) == 0:
            pop = self.initialise(N_init, 1, altern = {})
        S = [0, 0]
        N = []
        glc = []
        ac = []
        pop_state = []
        growth_rates = []
        timepoints = []
        strain_frequencies = []
        glucose = True # switch back and forth between acetate and glucose
        # Time keeping. Estimate how many switches must occur over time time
        for i in range(int(np.ceil(time/float(switch)))):
            # At the simulation end, the remaining time might be less than switch
            if timepoints:
                step = min(switch, time-timepoints[-1])
            else:
                step = switch
            if glucose:
                pop, n, g, a, tp, ps, sf, gr = self.simulate(pop, S, 20, 0, dil, step, timestep, verbose, True, save_state, True)
                if verbose:
                    print(" in glucose")
            else:
                pop, n, g, a, tp, ps, sf, gr = self.simulate(pop, S, 0, 40, dil, step, timestep, verbose, True, save_state, True)
                if verbose:
                    print(" in acetate")
            # Update Substrate and influx flag
            S = [g[-1], a[-1]]
            glucose = glucose != True
            N.extend(n)
            glc.extend(g)
            ac.extend(a)
            strain_frequencies.extend(sf)
            growth_rates.extend(gr)
            if save_state:
                pop_state.extend(ps)
            if timepoints:
                timepoints.extend([timepoints[-1] + t for t in tp]) # As tp gets reinitialised
            else:
                # As timepoints is initially empty
                timepoints.extend(tp)
        self.results["pop"] = pop
        self.results["N"] = N
        self.results["glc"] = glc
        self.results["ac"] = ac
        self.results["time"] = timepoints
        self.results["strain_frequencies"] = strain_frequencies
        if save_growth:
            self.results["growth_rates"] = growth_rates
        if save_state:
            self.results["pop_state"] = pop_state

    def save_instance(self, filename):
        """Saves instance in a pickle"""
        if os.path.isfile(self.results_directory+filename+'.pkl'):
            raise RuntimeError('File already exists')
        with open(self.results_directory+filename+'.pkl', 'w') as f:
            pickle.dump(self, f)
        return None
