#!/usr/bin/env python

import unittest
import warnings
import numpy as np
import population as pp
import modelanalysis as ma

"""Testing the functions in population.py and its analysis"""

# Parameter values as of 11/2/19
parameters = {

# Metabolism
"KE1_glc" : 0.1,    # Michaelis-Menten dissociation constant of E1 for glucose (mmol/L)
"KE2_ac" : 7.0,     # Michaelis-Menten dissociation constant of E2 for acetate (mmol/L)
"KE3_pep" : 0.3,     # Michaelis-Menten dissociation constant of E3 for PEP (mmol/g)
"KE4_pep" : 0.008,  # MWC dissociation constant of FBPase for PEP (mmol/g) Kotte: 0.3e-3 Works well with 0.008
"KE4_fbp" : 0.1, # MWC dissociation constant of FBPase for FBP (mmol/g) Kotte: 3e-6
"L_fbp" : 4e6,      # MWC kinetics allosteric constant (no units)
"n_fbp" : 4, # MWC cooperativity constant
"kE1_cat_s" : 100., # E1 turnover number (/s)
"kE2_cat_s" : 100,  # E2 turnover number (/s)
"kE3_cat_s" : 200, # E3 turnover number (/s)
"kE4_cat_s" : 200., # E4 turnover number (/s)

# Cra activity
"n_e" : 2,          # Hill coefficient for Cra (FBP binding) (no units)
"KCra_fbp" : 0.1,    # Cra Hill dissociation constant for FBP concentration (mmol/g) Kotte: 1.36e-3
"K_cra" : 25,       # Affinity of the E2 promoter for active Cra (copy number)
# Expression parameters
"nbE1" : 10,
"bsE1" : 500,
"nbE2_0" : 0,
"nbE2_1" : 50,
"bsE2" : 400,
"nbE3" : 20,
"bsE3" : 200,
"nbE4" : 10,
"bsE4" : 200,
"nbCra" : 5,
"bsCra" : 20,
# Protein degradation (per hour per protein)
"gamma": 0,
}

class TestGrowthandMetabolism(unittest.TestCase):

    def setUp(self):
        self.sim = pp.Simulation(parameters)

    def test_fructosebisphosphatase(self):
        # Once expected within-cell concentrations are known, make this function
        # test several concentrations of X as well
        FBPconc = np.linspace(0, 1e-6, 5)
        fluxes = self.sim.fructosebisphosphatase(2000, 0.5, FBPconc, self.sim.cellmass)
        sol =  np.array([0., 0.53640512, 0.99618103, 1.39465353, 1.743317])
        self.assertEqual(fluxes.all(), sol.all())

    def test_glcflux(self):
        Glcconc = np.linspace(0, 0.3, 5)
        fluxes = self.sim.glcflux(2000, Glcconc, self.sim.cellmass)
        sol = np.array([ 0., 3.5862527, 5.02075378, 5.79317744, 6.27594222])
        self.assertEqual(fluxes.all(), sol.all())

    def test_acflux(self):
        Acconc = np.linspace(0, 10, 5)
        fluxes = self.sim.acflux(2000, Acconc, self.sim.cellmass)
        sol = np.array([ 0., 2.0972238, 3.32060435, 4.12212954, 4.68791202])
        self.assertEqual(fluxes.all(), sol.all())

    def test_pepflux(self):
        pepconc = np.linspace(0, 3, 5)
        fluxes = self.sim.pepflux(2000, pepconc, self.sim.cellmass)
        sol = np.array([ 0.,  5.6924646, 6.6412087, 7.03186804, 7.24495495])
        self.assertEqual(fluxes.all(), sol.all())

    def test_Cra_synthesis(self):
        np.random.seed(0)
        # These functions could perhaps be better tested vs a negative
        # binomial distribution. This would also allow me to check if the
        # parameter settings are correct. This alternative test would be
        # some kind of statistical test (same kind of discrete distribution)
        self.sim.bsCra = 5
        self.sim.nbCra = 20
        Cra = self.sim.Cra_synthesis(self.sim.nbCra, self.sim.bsCra,.2, 1, self.sim.cellmass)
        self.assertEqual(Cra, 48)

    def test_E1_synthesis(self):
        np.random.seed(0)
        self.sim.nbE1 = 5
        self.sim.bsE1 = 20
        E1 = self.sim.E1_synthesis(0.2, 1, self.sim.cellmass)
        self.assertEqual(E1, 27)

    def test_E2_synthesis(self):
        np.random.seed(0)
        self.sim.nbE_0 = 5
        self.sim.nbE_1 = 20
        self.sim.bs = 100
        nb = self.sim.E2_expected_burst_number(10/self.sim.cellmass)
        E2 = self.sim.E2_synthesis(nb, 0.2, 1, self.sim.cellmass)
        self.assertEqual(E2, 2862)
        np.random.seed(0)

    def test_E3_synthesis(self):
        np.random.seed(0)
        self.sim.nbE3 = 5
        self.sim.bsE3 = 20
        E3 = self.sim.E3_synthesis(0.2, 1, self.sim.cellmass)
        self.assertEqual(E3, 27)

    def test_E4_synthesis(self):
        np.random.seed(0)
        self.sim.nbE4 = 5
        self.sim.bsE4 = 20
        E3 = self.sim.E4_synthesis(0.2, 1, self.sim.cellmass)
        self.assertEqual(E3, 27)

    def test_cell_growth(self):
        # This function is a wrapper for the reaction functions
        # It determines the relation between metabolite flux (FBP flux)
        # and growth rate
        # It calculates the change in metabolite concentrations using the fluxes
        # from the reaction functions. These must follow stoicheometric
        # constraints

        # Check that stoicheometry is conserved
        np.random.seed(0)
        pop_old = self.sim.initialise(1)
        pop_old[0, 1] = 1e-3
        dpopdt, dglcdt, dacdt, J_glc, J_ac, J_pep, J_fbp, gr = self.sim.cell_growth(pop_old, 1, 20., 10., return_all=True)
        gr = dpopdt[0,0] / pop_old[0,0]
        J_glc = dglcdt * self.sim.V / pop_old[0,0]
        J_ac = dacdt * self.sim.V / pop_old[0,0]
        FBP = pop_old[0,1]
        dFBPdt = dpopdt[0,1]
        Jfbp_expected = -(dFBPdt + gr * FBP - J_glc - 1/2. * J_pep)
        expected_yield = gr / Jfbp_expected
        self.assertAlmostEqual(0.0896, expected_yield[0], places=4)

    def test_protein_synthesis(self):
        # This function is a wrapper for the protein synthesis functions
        # Within it the amount of Cra activity is calculated
        np.random.seed(0)
        cell_old = np.array([self.sim.cellmass, 0, 0, 240, 2721, 3391, 6340, 7800, 0.2, self.sim.nbCra, self.sim.bsCra])
        cell_new = self.sim.protein_synthesis(cell_old, 0.9, 1/6.)
        cell_exp = np.array([self.sim.cellmass, 0, 0,258,4121, 7963, 7174, 7922, 0.2, self.sim.nbCra, self.sim.bsCra])
        self.assertTrue((cell_exp == cell_new).all())

    def test_protein_synthesis_cra_nonstochastic(self):
        # Test that cra_stochastic = False switches off stochastic Cra production
        # ... and that the number of cra molecules increases at the same rate as
        # the biomass
        np.random.seed(0)
        cell_old = np.array([self.sim.cellmass, 0, 0, 240, 2721, 3391, 6340, 7800, 0.2, self.sim.nbCra, self.sim.bsCra])
        cell_new = self.sim.protein_synthesis(cell_old, 0.9, 1/6., cra_stochastic=False)
        self.assertTrue(cell_new[3] == 100)
        # At end of cell cycle the cell should contain twice as much Cra
        cell_old[0] = 2*self.sim.cellmass
        cell_new = self.sim.protein_synthesis(cell_old, 0.9, 1/6., cra_stochastic=False)
        self.assertTrue(cell_new[3] == 200)

    def test_protein_synthesis_pass_on_gamma(self):
        # Test that E2 is produced at the rate that it should be if gamma > 0
        # passed to E2 directly
        np.random.seed(0)
        self.sim.gamma=10
        CraA = self.sim.cra_activity(100, 0.1, self.sim.KCra_fbp)
        nbe2 = self.sim.E2_expected_burst_number(CraA/self.sim.cellmass)
        e1 = self.sim.E1_synthesis(0.1, 1/60., self.sim.cellmass)
        e2 = self.sim.E2_synthesis(nbe2, (0.1+self.sim.gamma), 1/60., self.sim.cellmass)
        # protein synthesis
        np.random.seed(0)
        cell_old = np.array([self.sim.cellmass, 0.1, 0, 100, 0, 0, 0, 0, self.sim.KCra_fbp, self.sim.nbCra, self.sim.bsCra])
        cell_new = self.sim.protein_synthesis(cell_old, 0.1, 1/60., cra_stochastic=False)
        self.assertEqual(e2, cell_new[5])

    def test_protein_degradation(self):
        np.random.seed(0)
        self.sim.gamma = 0.1
        cell_old = np.array([self.sim.cellmass, 0, 0, 240, 2721, 3391, 6340, 7800, 0.2, self.sim.nbCra, self.sim.bsCra])
        cell_new = self.sim.protein_degradation(cell_old, 1)
        # Protein concentrations should be less afterwards
        self.assertTrue(cell_new[5] == 3066)


class TestPopulation(unittest.TestCase):

    def setUp(self):
        self.sim = pp.Simulation(parameters)

    def test_initialise(self):
        # No cells should be the same
        pop = self.sim.initialise(100)
        pop_set = {tuple(cell) for cell in pop}
        self.assertEqual(len(pop_set), len(pop))
        # Make sure there are no np.nans
        self.assertFalse(np.isnan(pop).any())

    def test_cell_division_population(self):
        np.random.seed(0)
        pop_old = self.sim.initialise(10)
        pop_old[:,0] += 1e-13
        pop_new = self.sim.cell_division(pop_old)
        # Three cells have divided [1, 7, 8] : Population increased by three
        self.assertEqual(len(pop_new), len(pop_old) + 3 )
        # Cells that are too small (less than 2 * self.sim.cellmass) should not divide
        notdivided_old = np.sort(pop_old[pop_old[:,0] < 2 * self.sim.cellmass])
        notdivided_new = np.sort(pop_new[:-6])
        self.assertTrue((notdivided_old == notdivided_new).all())
        # Make sure there are no individual cells
        new_set = {tuple(cell) for cell in pop_new}
        self.assertEqual(len(pop_new), len(new_set))
        # Make sure there are no np.nans
        self.assertFalse(np.isnan(pop_new).any())

    def test_cell_division_individual(self):
        # Cells that have divided:
        # Test with a single cell
        np.random.seed(0)
        parent = self.sim.initialise(1)
        parent[:,0] += 2e-13
        offspring = self.sim.cell_division(parent)
        # (i) Should have their mass halved
        self.assertEqual(offspring[0,0], 0.5 * parent[0,0])
        self.assertEqual(offspring[1,0], 0.5 * parent[0,0])
        # (ii) Keep constant metabolite concentrations
        # (only the number of moles halves)
        self.assertEqual(offspring[0,1], parent[0,1])
        self.assertEqual(offspring[1,1], parent[0,1])
        self.assertEqual(offspring[0,2], parent[0,2])
        self.assertEqual(offspring[1,2], parent[0,2])
        # (iii) The sum of the proteins in the two offspring cells should
        # equal that of the parent cell
        self.assertEqual(np.sum(offspring[:,3]), parent[0,3])
        self.assertEqual(np.sum(offspring[:,4]), parent[0,4])
        self.assertEqual(np.sum(offspring[:,5]), parent[0,5])
        self.assertEqual(np.sum(offspring[:,6]), parent[0,6])

    def test_cell_efflux(self):
        # Cells should not be sampled more than twice
        # With increasing time or dilution rate, more cells should be washed out
        # If there are no cells, simply return empty array
        np.random.seed(0)
        pop_old = self.sim.initialise(10)
        pop_new = self.sim.cell_efflux(pop_old, 0.2, 0.5)
        # The number of cells afterwards should be less or equal
        self.assertTrue(len(pop_new) <= len(pop_old) )
        # No cells should be the same
        new_set = {tuple(cell) for cell in pop_new}
        self.assertEqual(len(pop_new), len(new_set))

    def test_chemostat_growth(self):
        # make sure that chemostat results in growth
        np.random.seed(0)
        pop = self.sim.initialise(1)
        self.sim.simulate(pop, [20,0], 20, 0, 0.1, 5/60., 1/60., False, False, False)
        mass = self.sim.results['pop'][0,0]
        fbp = self.sim.results['pop'][0,1]
        E4 = self.sim.results['pop'][0,7]
        self.assertAlmostEqual(mass, 4.6557531e-13, places=4)
        self.assertAlmostEqual(fbp, 2.4435241, places=4)
        self.assertEqual(int(E4), 2336)

    def test_chemostat_protein_degradation(self):
        np.random.seed(0)
        self.sim.gamma=0.1
        pop = self.sim.initialise(1)
        npop, S, gr = self.sim.chemostat(pop,[20,0], 20, 0, 0.1, 1/60.)
        self.assertEqual(npop[0,5],629)

    def test_chemostat_individuality(self):
        # Make sure that two cells don't behave exactly the same
        np.random.seed(0)
        pop_state = []
        S = [0,0]
        pop = self.sim.initialise(2)
        for i in range(100):
            pop, S, gr = self.sim.chemostat(pop, S, 10, 1, 0.2, 1/60.)
            pop_set = {tuple(cell) for cell in pop}
            self.assertEqual(len(pop), len(pop_set))

class TestModelAnalysis(unittest.TestCase):

    def setUp(self):
        self.Nmodel = ma.Numericalanalysis(pp.parameters)
        self.Smodel = ma.Symbolicanalysis(pp.parameters)

    def test_equivalent_behaviour(self):
        # Make sure that Numeric and Symbolic analysis give same results
        # Relative error tolerance should be
        with warnings.catch_warnings():
            # Gives a RuntimeWarning, can be ignored
            warnings.simplefilter("ignore")
            eq = self.Nmodel.find_equilibrium(xspace=np.logspace(-10,2,3), verbose=False)
            res = self.Smodel.find_equilibrium(xspace=np.logspace(-10,2,3), verbose=False)
        # Compare equilibrium acetate concentrations (most computational steps)
        aceq = np.array(res['acstable']).astype(float)
        diff = aceq - eq[:,0]
        self.assertTrue((abs(diff) < 1e-16).all())



if __name__ == "__main__":
    unittest.main()
