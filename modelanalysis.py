#!/usr/bin/env python

"""Find the equilibria and their stability"""

import os
import sys
import sympy
import mpmath
import numpy as np
sys.path.append(os.getcwd())
import population as pp
import scipy.optimize as spi
from mpmath.libmp.libhyper import NoConvergence

try:
    # in case import fails on cluster
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from seaborn import blend_palette
except TypeError:
    pass

# Pretty printing
sympy.init_printing(use_unicode=True)

class Numericalanalysis:

    def __init__(self, parameters):
        self.sim = pp.Simulation(parameters)
        self.E3 = self.sim.nbE3 * self.sim.bsE3
        self.E4 = self.sim.nbE4 * self.sim.bsE4
        self.Cra = self.sim.nbCra * self.sim.bsCra
        self.c = 0.0896
        self.x = 0 # pep concentration
        self.y = 0 # fbp concentration

    def E2_copy_number(self, y):
        """Calculate the number of E2 per cell"""
        CraA=self.sim.cra_activity(self.Cra, y, self.sim.KCra_fbp)
        return self.sim.E2_expected_burst_number(CraA/self.sim.cellmass)*self.sim.bsE2

    def dxdt(self, ac):
        """Get out ac from given self.x and self.y"""
        J_ac = self.sim.acflux(self.E2_copy_number(self.y), ac, self.sim.cellmass)
        J_pep = self.sim.pepflux(self.E3, self.x, self.sim.cellmass)
        J_fbp = self.sim.fructosebisphosphatase(self.E4, self.x, self.y, self.sim.cellmass)
        return 0.5 * J_ac - J_pep - self.c*J_fbp*self.x

    def dydt(self,y):
        """Get out y from given self.x"""
        J_pep = self.sim.pepflux(self.E3, self.x, self.sim.cellmass)
        J_fbp = self.sim.fructosebisphosphatase(self.E4, self.x, y, self.sim.cellmass)
        dydt = 0.5 * J_pep - J_fbp - self.c*J_fbp*y
        return dydt

    def growth(self, x, y):
        """Convenience function to calculate the growth rate"""
        return self.c*self.sim.fructosebisphosphatase(self.E4, x, y, self.sim.cellmass)

    def find_equilibrium_point(self, x):
        """For a given point x, find the corresponding y and ac values. Raises
        ValueError if no equilibrium exists"""
        self.x = x
        # Approximate solution for y
        tmpy = spi.fsolve(self.dydt, 0.1) # Anything will do
        # if tmpy is negative, try a lower root
        k = 2
        while tmpy[0] < 0 and k < 100:
            tmpy = spi.fsolve(self.dydt, 10 ** -k)
            k += 1
        ysol = mpmath.findroot(self.dydt, [float(tmpy[0])], method='mnewton')
        self.y = float(ysol.real)
        # Approximate solution for ac
        tmpac = spi.fsolve(self.dxdt, 0.1) # Anything will do
        k = 2
        while tmpac < 0 and k < 100:
            tmpac = spi.fsolve(self.dxdt, 10 ** -k)
            k += 1
        self.y = ysol
        acsol = mpmath.findroot(self.dxdt, [float(tmpac[0])], method='mnewton')
        sol = [float(acsol.real), x, float(ysol.real)]
        if any([i < 0 for i in sol]):
            raise ValueError
        else:
            return sol


    def find_equilibrium(self, xspace=np.logspace(-10,2,500), verbose=True):
        """Find equilibrium points for the values in xspace (if they exist)"""
        eqpoints=[]
        for v in xspace:
            try:
                point = self.find_equilibrium_point(v)
            except ValueError:
                if verbose:
                    print "No solution found for pep = {}".format(v)
                pass
            else:
                eqpoints.append(point)
        return np.array(eqpoints)

    def growth(self, x, y):
        """Calculate growth rate given pep and fbp concentrations x and y"""
        return self.c*self.sim.fructosebisphosphatase(self.E4, x, y, self.sim.cellmass)


class Symbolicanalysis:

    def __init__(self, parameters):
        """Uses parameters to create a sympy ODE system for analysis"""

        self.sim = pp.Simulation(parameters)
        self.E3 = self.sim.nbE3 * self.sim.bsE3
        self.E4 = self.sim.nbE4 * self.sim.bsE4

        # define parameters
        self.Cra = self.sim.nbCra * self.sim.bsCra
        self.K_cra_fbp = self.sim.KCra_fbp
        self.K_cra_dna = self.sim.K_cra
        self.n = self.sim.n_e
        self.n_fbp = int(self.sim.n_fbp)
        self.k_cat_E2 = self.sim.kE2_cat_s * 3600 / (self.sim.cellmass * 6.023e20)
        self.V_max_pep = self.E3 * self.sim.kE3_cat_s * 3600 / (self.sim.cellmass * 6.023e20)
        self.V_max_fbp = self.E4 * self.sim.kE4_cat_s * 3600 / (self.sim.cellmass * 6.023e20)
        self.K_ac, self.K_pep, self.K_fbp_fbp, self.K_fbp_pep = self.sim.KE2_ac, self.sim.KE3_pep, self.sim.KE4_fbp, self.sim.KE4_pep
        L = self.sim.L_fbp
        self.c = 0.0896

        #ac = 5.5
        # define symbols x (for pep), y (for fbp), z (for E2) ac for extracellular acetate concentration
        self.x, self.y, self.z, self.ac = sympy.symbols('x y z ac', real=True, negative=False)

        # Define equations
        self.J_ac = self.k_cat_E2 * self.z * self.ac / (self.ac + self.K_ac)
        self.J_pep = self.V_max_pep * self.x / (self.x + self.K_pep)
        self.J_fbp = (self.V_max_fbp * (self.y / self.K_fbp_fbp) * (1 + self.y / self.K_fbp_fbp) ** (self.n_fbp-1)) / ( (1 + self.y / self.K_fbp_fbp)**self.n_fbp + L * (1 + self.x/self.K_fbp_pep)**-self.n_fbp )
        def active_cra():
            CraA = self.Cra * (1 - self.y**self.n / (self.y**self.n + self.K_cra_fbp**self.n ) ) / self.sim.cellmass
            return CraA

        self.h_y = self.sim.nbE2_1 * self.sim.bsE2 * active_cra() / (active_cra() + self.K_cra_dna)

        # Define differential equations
        self.dxdt = 0.5 * self.J_ac - self.J_pep - self.c*self.J_fbp*self.x
        self.dydt = 0.5 * self.J_pep - self.J_fbp - self.c*self.J_fbp*self.y
        self.dzdt = self.c * self.J_fbp * (self.h_y - self.z)

        # ODE system in matrix form
        self.ODEsys = sympy.Matrix([self.dxdt, self.dydt, self.dzdt])

        return None

    def E2_copy_number(self, y):
        """Calculate the number of E2 per cell"""
        CraA=self.sim.cra_activity(self.Cra, y, self.K_cra_fbp)
        return self.sim.E2_expected_burst_number(CraA/self.sim.cellmass)*self.sim.bsE2

    def find_eigenvalues(self, AC=0, X=0, Y=0, Z=0):
        """Returns the real eigenvalues at point X,Y,Z given acetate conc AC"""
        # Find jacobian
        self.jac = self.ODEsys.jacobian([self.x,self.y,self.z])
        jmat = self.jac.subs([(self.x, X), (self.y, Y), (self.z, Z), (self.ac, AC)])
        eigreal = np.linalg.eig(np.array(jmat).astype(float))[0].real
        return eigreal

    def sort_equilibrium_points(self, points):
        """Sorts a set of equilibrium points into stable and unstable points.
        Each point must be in the form [AC, X, Y, Z]"""
        stable = []
        unstable = []
        for p in points:
            eigreal = self.find_eigenvalues(p[0], p[1], p[2], p[3])
            if (eigreal <= 0).all():
                stable.append(p)
            else:
                unstable.append(p)
        return np.array(stable), np.array(unstable)

    def find_equilibrium(self, xspace=np.logspace(-10,2,500), verbose=True):
        """Find equilibrium by sequentially solving the nullclines of the ODE system"""
        # Find jacobian
        jac = self.ODEsys.jacobian([self.x, self.y, self.z])

        # Finding steady states (this does not work/takes far too long)?
        dxdt_null = sympy.Eq(self.dxdt,0)
        dydt_null = sympy.Eq(self.dydt,0)
        dzdt_null = sympy.Eq(self.dzdt,0)

        # Going backwards: Starting from x, determine y. From y, determine z. Use x,y,z
        # to determine ac. Check if position is at equilibrium. Investigate
        # stability using Jacobian Matrix

        acstable = []
        acunstable = []
        ystable = []
        yunstable = []
        xstable=[]
        xunstable=[]
        trajectories = []
        equilibria = []
        for x_eq in xspace:
            try:
                dydt_null_x = dydt_null.subs(self.x,x_eq)
                y_eq = sympy.solve(dydt_null_x, rational=False)
                dzdt_null_x_y = dzdt_null.subs([(self.y, y_eq[0]), (self.x, x_eq)])
                z_eq = sympy.solve(dzdt_null_x_y, rational=False)
                dxdt_null_x_y_z = dxdt_null.subs([(self.x,x_eq),(self.y,y_eq[0]),(self.z,z_eq[0])])
                ac_eq = sympy.solve(dxdt_null_x_y_z, rational=False)
                # Check if the solution is indeed close to equilibrium (within margin of error)
                dxdt_eq = self.dxdt.subs([(self.x,x_eq),(self.y,y_eq[0]),(self.z,z_eq[0]),(self.ac,ac_eq[0])])
                dydt_eq = self.dydt.subs([(self.x,x_eq),(self.y,y_eq[0])])
                dzdt_eq = self.dzdt.subs([(self.x,x_eq),(self.y,y_eq[0]),(self.z,z_eq[0])])
            except (TypeError, NoConvergence) as e:
                if verbose:
                    print "No solution at {}".format(x_eq)
                next
            else:
                if verbose:
                    if abs(dxdt_eq) < 1e-15 and abs(dydt_eq) < 1e-15 and abs(dzdt_eq) < 1e-15:
                        print "Equilibrium point"
                    else:
                        print "Something's wrong!"
                # Determine stability with the jacobian matrix
                # If all real parts of the eigenvalues are negative, the equilibrium is stable
                eqmat = jac.subs([(self.x,x_eq),(self.y,y_eq[0]),(self.z,z_eq[0]),(self.ac, ac_eq[0])])
                eigenvalues_real = np.linalg.eig(np.array(eqmat).astype(float))[0].real
                if verbose:
                    print eigenvalues_real
                if all([e < 0 for e in eigenvalues_real]):
                    xstable.append(x_eq)
                    ystable.append(y_eq[0])
                    acstable.append(ac_eq[0])
                else:
                    xunstable.append(x_eq)
                    yunstable.append(y_eq[0])
                    acunstable.append(ac_eq[0])
                equilibria.append([ac_eq[0], x_eq, y_eq[0], z_eq[0]])

        return {'equil': equilibria, 'acstable': acstable, 'acunstable': acunstable,
                'xstable': xstable, 'xunstable': xunstable, 'ystable': ystable,
                'yunstable': yunstable}

    def plot_results(self, results, save_file='../Results/equilibrium/ODE_stability.tiff'):
        xstable = results['xstable']
        ystable = results['ystable']
        xunstable = results['xunstable']
        yunstable = results['yunstable']
        acstable = results['acstable']
        acunstable = results['acunstable']
        fig, ax = plt.subplots(1,2, figsize=[16,6])
        ax[0].plot(xstable,ystable,'ko', label='stable')
        ax[0].plot(xunstable,yunstable, 'ro', label='unstable')
        ax[0].set_xscale('log')
        ax[0].set_xlabel('pep concentration (mmol/g)', fontsize=14)
        ax[0].set_ylabel('fbp concentration (mmol/g)', fontsize=14)
        ax[0].tick_params(labelsize=14, axis="both")
        ax[0].legend(frameon=False, loc="upper center", fontsize=14)
        ax[0].annotate("A", xy=(0,1), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", fontsize=25)
        growthstable = [self.c*self.sim.fructosebisphosphatase(self.E4,xval,yval,self.sim.cellmass) for (xval,yval) in zip(xstable,ystable)]
        growthunstable = [self.c*self.sim.fructosebisphosphatase(self.E4,xval,yval,self.sim.cellmass) for (xval,yval) in zip(xunstable,yunstable)]
        ax[1].plot(acstable, growthstable, 'ko')
        ax[1].plot(acunstable, growthunstable, 'ro')
        ax[1].set_xlim(0,12)
        ax[1].set_ylim(0,0.36)
        ax[1].set_xlabel('acetate concentration (mM)', fontsize=14)
        ax[1].set_ylabel('growth rate (per hour)', fontsize=14)
        ax[1].tick_params(labelsize=14, axis="both")
        ax[1].fill_between([float(np.min(acunstable))-0.1, float(np.max(acunstable))+0.1],
            np.repeat(0,2),
            np.repeat(0.4,2),
            facecolor="silver")
        ax[1].annotate("B", xy=(0,1), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", fontsize=25)
        if save_file:
            fig.savefig(save_file, dpi=600, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def find_nullclines(self):
        """Find nullclines for dydt and dxdt at a given ac and equilibrium z"""
        dxdt_null = sympy.Eq(self.dxdt,0)
        dydt_null = sympy.Eq(self.dydt,0)
        dzdt_null = sympy.Eq(self.dzdt,0)
        # Here, define an alternative version of dxdt, in which z is always at the equilibrium
        # given by h_y
        J_ac_X = self.k_cat_E2 * self.h_y * self.ac / (self.ac + self.K_ac)
        dXdt = 0.5*J_ac_X - self.J_pep - self.c*self.J_fbp*self.x
        dXdt_null = sympy.Eq(dXdt,0)
        ac_nc = 5.5
        dXdt_y_vals=[]
        dXdt_x_vals=[]
        dydt_y_vals=[]
        dydt_x_vals=[]
        for x_nc in np.logspace(-5,-0.1,500):
            y_dydt_nc = dydt_null.subs(self.x,x_nc)
            y_dydt_val = sympy.solve(y_dydt_nc, rational=False)
            dXdt_null_ac = dXdt_null.subs(self.ac,ac_nc)
            y_dXdt_nc = dXdt_null_ac.subs(self.x,x_nc)
            y_dXdt_val = sympy.solve(y_dXdt_nc, rational=False)
            if y_dXdt_val:
                dXdt_y_vals.extend(y_dXdt_val)
                dXdt_x_vals.append(x_nc)
            if y_dydt_val:
                dydt_y_vals.extend(y_dydt_val)
                dydt_x_vals.append(x_nc)
            sys.stdout.write('#')
            sys.stdout.flush()
        nullclines = {'dXdt_x_vals': dXdt_x_vals, 'dXdt_y_vals': dXdt_y_vals,
            'dydt_y_vals': dydt_y_vals, 'dydt_x_vals': dydt_x_vals}
        return nullclines

    def draw_nullclines(self, nullclines, save_file=None):
        """Plot nullclines and behaviour of ODE system around the nullclines"""
        ac_nc = 5.5
        dXdt_y_vals=nullclines['dXdt_y_vals']
        dXdt_x_vals=nullclines['dXdt_x_vals']
        dydt_y_vals=nullclines['dydt_y_vals']
        dydt_x_vals=nullclines['dydt_x_vals']
        # These are the indicies over which arrows will be drawn
        start, stop, step = 300, 445, 10
        dydt_y_vals = np.array(dydt_y_vals).astype(float)
        dydt_x_vals = np.array(dydt_x_vals).astype(float)
        dXdt_y_vals = np.array(dXdt_y_vals).astype(float)
        dXdt_x_vals = np.array(dXdt_x_vals).astype(float)
        xval,yval = np.meshgrid(dydt_x_vals[start:stop:step], np.linspace(-0.05,0.05,25))
        fig, axes = plt.subplots(1,2,figsize=[16,6])
        axes[0].plot(dXdt_x_vals, dXdt_y_vals, color='red', label='pep nullcline')
        axes[0].plot(dydt_x_vals, dydt_y_vals, color='black', label='fbp nullcline')
        #axes[0].plot(xval, dydt_y_vals[start:stop:step]+yval, 'k,')
        axes[0].fill_between(dydt_x_vals[start:stop:step], dydt_y_vals[start:stop:step]-0.05, dydt_y_vals[start:stop:step]+0.05, color='grey')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('pep concentration (mmol/g)', fontsize=14)
        axes[0].set_ylabel('fbp concentration (mmol/g)', fontsize=14)
        axes[0].tick_params(labelsize=14, axis='both')
        axes[0].legend(frameon=False, fontsize=14)
        axes[0].set_xlim(min(dydt_x_vals), max(dydt_x_vals))
        axes[0].set_ylim(0,1.1)
        axes[0].annotate("A", xy=(-0.1,1.15), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", fontsize=25)

        zval = self.sim.E2_expected_burst_number(self.sim.cra_activity(self.Cra, yval+dydt_y_vals[start:stop:step], self.sim.KCra_fbp)/self.sim.cellmass)*self.sim.bsE2
        dXdt_quiver = 1/2. * self.sim.acflux(zval, ac_nc, self.sim.cellmass) - self.sim.pepflux(self.E3, xval, self.sim.cellmass) - self.c * self.sim.fructosebisphosphatase(self.E4, xval, yval+dydt_y_vals[start:stop:step], self.sim.cellmass) * xval
        dYdt_quiver = 1/2. * self.sim.pepflux(self.E3, xval, self.sim.cellmass) - self.sim.fructosebisphosphatase(self.E4, xval,yval+dydt_y_vals[start:stop:step], self.sim.cellmass) - self.c * self.sim.fructosebisphosphatase(self.E4,xval,yval+dydt_y_vals[start:stop:step],self.sim.cellmass) * (yval+dydt_y_vals[start:stop:step])
        U = dXdt_quiver / np.sqrt(dXdt_quiver**2 + dYdt_quiver**2)
        V = dYdt_quiver / np.sqrt(dXdt_quiver**2 + dYdt_quiver**2)
        axes[1].quiver(xval,yval, U, V, headwidth=5, pivot='mid')#, pivot='tip', angles='xy', scale=50)
        axes[1].plot(dXdt_x_vals,[u-v for u,v in zip(dXdt_y_vals,dydt_y_vals[:len(dXdt_y_vals)])], color='red')
        axes[1].axhline(0, color='black')
        axes[1].set_xscale('log')
        #axes[1].yscale('symlog')
        axes[1].set_xlim(0.008,0.22)
        axes[1].set_ylim(-0.05,0.05)
        axes[1].set_ylabel('fbp concentration - nullcline (mmol/g)', fontsize=14)
        axes[1].set_xlabel('pep concentration (mmol/g)', fontsize=14)
        axes[1].tick_params(labelsize=14, axis='both')
        axes[1].annotate("B", xy=(-0.1,1.15), xycoords="axes fraction",
                        xytext=(5,-5), textcoords="offset points",
                        ha="left", va="top", fontsize=25)
#        plt.tight_layout()
        if save_file:
            fig.savefig(save_file, dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def draw_moving_nullcline(self, acetate=np.logspace(0,1,100), xspace=np.logspace(-10,1,500), yspace=np.linspace(0,0.5,500)):
        """Produce an animation of moving pep nullcline with changing acetate concentration"""
        x,y = np.meshgrid(xspace, yspace)
        for ac in acetate:
            plt.contour(x,y,0.5*self.sim.pepflux(self.E3,x,self.sim.cellmass)- self.sim.fructosebisphosphatase(self.E4,x,y,self.sim.cellmass)- self.sim.fructosebisphosphatase(self.E4,x,y,self.sim.cellmass)*self.c*y,[0], colors='red')
            plt.contour(x,y,0.5*self.sim.acflux(self.E2_copy_number(y),ac,self.sim.cellmass)-self.sim.pepflux(self.E3,x,self.sim.cellmass) - self.sim.fructosebisphosphatase(self.E4,x,y,self.sim.cellmass)*self.c*x,[0])
            plt.xscale('log')
            plt.xlabel('pep concentration log10(mmol/g)')
            plt.ylabel('fbp concentration (mmol/g)')
            plt.title(ac)
            plt.show(block=False)
            plt.pause(0.01)
            plt.cla()
            plt.clf()
        plt.close()

def check_bistability(stable, unstable):
    """Simple test. If there is an unstable region between two
    stable regions, the model is bistable"""
    if len(stable) == 0 or len(unstable) == 0:
        return False
    else:
        unst_min = np.min(unstable[:,1])
        unst_max = np.max(unstable[:,1])
        st_min = np.min(stable[:,1])
        st_max = np.max(stable[:,1])
        # There should be no stable point within the range of the unstable region
        if ((stable[:,1] > unst_min) & (stable[:,1] < unst_max)).any():
            return False
        else:
            # The unstable region should lie between two stable regions
            if st_max > unst_max and st_min < unst_min:
                return True
            else:
                return False

def Cra_number_effects(cra=range(10,101,10), parameters=pp.parameters, nsteps=1000):
    """Calculate how changing the Cra copy number affects equilibrium"""
    Nmodel = Numericalanalysis(parameters)
    # Make Smodel instance for determining stability
    Smodel = Symbolicanalysis(parameters)
    equilibriums = {}
    for i in range(len(cra)):
        Nmodel.Cra = cra[i]
        eq = Nmodel.find_equilibrium(xspace=np.logspace(-10,2,nsteps), verbose=False)
        equilibriums[cra[i]] = eq
    return Nmodel, Smodel, equilibriums

def plot_Cra_number_effects(directory, Nmodel, equilibriums, parameters=pp.parameters, save=True):
    parameters['nbCra'] = 1
    fig, ax = plt.subplots(figsize=[8,6])
    cra = sorted(equilibriums.keys())
    col = blend_palette(['black','red'], len(cra))
    cmap = blend_palette(['black','red'], len(cra), as_cmap=True)
    norm = colors.BoundaryNorm(cra, cmap.N)
    dummy=ax.scatter(cra, cra, c=cra, cmap=cmap, norm=norm)
    ax.cla()
    for i in range(len(cra)):
        eq = equilibriums[cra[i]]
        # The output of Nmodel does not include Z, that is the E2 copy numbers
        Nmodel.Cra = cra[i]
        parameters['bsCra'] = cra[i]
        Smodel = Symbolicanalysis(parameters)
        eq3 = [[a,x,y,Nmodel.E2_copy_number(y)] for [a,x,y] in eq]
        stable, unstable = Smodel.sort_equilibrium_points(eq3)
        gr_stable = Nmodel.growth(stable[:,1],stable[:,2])
        if len(unstable) > 0:
            gr_unstable = Nmodel.growth(unstable[:,1],unstable[:,2])
            ax.plot(unstable[:,0], gr_unstable, label=str(cra[i]), color=col[i], linestyle='dotted')
            # Split stable growth rates into high and low
            highindex = np.argwhere(gr_stable > gr_unstable.max())
            lowindex = np.argwhere(gr_stable < gr_unstable.min())
            ax.plot(stable[highindex,0], gr_stable[highindex], label=str(cra[i]), color=col[i], linestyle='solid')
            ax.plot(stable[lowindex,0], gr_stable[lowindex], label=str(cra[i]), color=col[i], linestyle='solid')
        else:
            ax.plot(stable[:,0], gr_stable, label=str(cra[i]), color=col[i], linestyle='solid')
    ax.set_xlim(0,60)
    ax.set_ylim(0,0.45)
    ax.set_xlabel('Acetate concentration (mM)', fontsize=14)
    ax.set_ylabel('Growth rate (per hour)', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    cbar=fig.colorbar(dummy, ticks=cra)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Cra number per cell", fontsize=14)
    plt.annotate("A", xy=(0,1), xycoords="axes fraction",
                    xytext=(5,-5), textcoords="offset points",
                    ha="left", va="top", fontsize=25)
    fig.savefig(directory+'Growth_rates_Cra_number.tiff', dpi=600, bbox_inches='tight')
    plt.close()

def main(parameters, xspace=np.logspace(-10,2,500), plot=True):
    """Find equilibrium point, determine their stability, and plot"""
    Nmodel = Numericalanalysis(parameters)
    eq = Nmodel.find_equilibrium(verbose=False, xspace=xspace)
    Smodel = Symbolicanalysis(parameters)
    # The output of Nmodel does not include Z, that is the E2/Ai copy numbers
    eq3 = [[a,x,y,Nmodel.E2_copy_number(y)] for [a,x,y] in eq]
    stable, unstable = Smodel.sort_equilibrium_points(eq3)
    if plot:
        plot_equilibrium(stable, unstable, Nmodel)
    return eq3, stable, unstable
