#Importing necessary libraries
import numpy as np 
import matplotlib.pyplot as plt
import scienceplots # Just for aesthetic purposes
import pandas as pd
import numpy.linalg as la
import sympy as sp
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import fsolve
import argparse

# To see the outputs in latex format, we use:
from sympy.interactive import printing
printing.init_printing(use_latex = True)

# Let's use an specific style for the plots!
plt.style.use(['science','notebook','grid'])
# Set the font size globally
plt.rcParams.update({
    'font.size': 10,        # Controls default text size
    'axes.titlesize': 12,   # Controls title size
    'axes.labelsize': 11,   # Controls x and y labels size
    'xtick.labelsize': 10,  # Controls x-tick labels size
    'ytick.labelsize': 10,  # Controls y-tick labels size
    'legend.fontsize': 10   # Controls legend font size
})


# Let's define some global constants
C = 63197.8 # [AU/year]
G = (6.67428e-11)*((1/(1.49598e11))**3)*(1.98842e30)*((3.154e7)**2) #[AU^3/M_sun year^2]
G = 4*(np.pi**2) # [AU^3/M_sun year^2]



class TwoBodyProblem:
    """
    Class to solve the two body problem.
    """
    def __init__(self, M, a, e):
        """
        Initialize the two body problem.

        Parameters
        ----------
        M, a, e
        """
        self.M = M
        self.a = a
        self.e = e
        
        # Let's define some fixed variables
        self.G = 4*(np.pi**2)
        self.T = 2*np.pi*np.sqrt((a**3)/(G*self.M)) #[years]
        self.R_s = (2*G*self.M)/(C**2)
        self.r0 = np.array([0, a*(1-e)])
        self.v0 = np.array([-np.sqrt((G*self.M/a)*((1+e)/(1-e))),0])
    
    def plot_grid(self, save=False):
        """
        
        """
        a = self.a
        R_s = self.R_s
        r0 = self.r0


        # Create plot
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(-2 * a, 2 * a)
        ax.set_ylim(-2 * a, 2 * a)
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title("Initial Two-Body Problem Setup")
        # Plot central black hole and Schwarzschild radius
        ax.scatter(0, 0, color='k', s=50, label="Black Hole")
        schwarzschild_circle = plt.Circle((0, 0), R_s, color='r', fill=False, linestyle='dashed', label="Schwarzschild Radius")
        ax.add_patch(schwarzschild_circle)
        # Plot initial position of orbiting body
        ax.scatter(r0[0], r0[1], color='b', s=25, label="Planet")
        ax.legend()
        if save:
            plt.savefig("initial_TBP.png")
        plt.show()



class Integrators:
    """
    This class contains the integrators for the two-body problem.
    """
    def __init__(self, N, dt, correction, two_body_instance):
        """
        
        """
        self.N = N # Number of orbits
        self.dt = dt
        self.correction = correction
        self.two_body_instance = two_body_instance
        self.T = self.two_body_instance.T
        self.t_span = [0, N*self.T]
        

    @staticmethod
    def slope(t, s, correction, M):
        """
        u is actual state
        u[0] is r
        u[1] is v
        """

        r = np.linalg.norm(s[0])
        L = np.linalg.norm(np.cross(s[0],s[1]))
        if correction:
            term1 = -(G*M/(r**3))*(1+(3*((L/(r*C))**2)))
            
        else:
            term1 = -(G*M/(r**3))            
        slope = np.array([s[1],term1*s[0]])
 
        return slope
    
    @staticmethod
    def slope_scipy(t, s0_flat, correction, M):
        """
        
        """
        r = s0_flat[:2]  # position components [x, y]
        v = s0_flat[2:]  # velocity components [vx, vy]
        
        r_norm = np.linalg.norm(r)
        L = np.linalg.norm(np.cross(r, v))
        
        if correction:
            term1 = -(G*M/(r_norm**3))*(1+((3*(L/(r_norm*C))**2)))
        else:
            term1 = -(G*M/(r_norm**3))
        
        slope = np.concatenate([v, term1*r])
        return slope 

    def trapezoidal(self):
        """

        """
        f=self.slope
        t_span = self.t_span
        dt = self.dt
        
        s0 = self.two_body_instance.s0
        M = self.two_body_instance.M

        # Initialize time array
        t_axis = np.arange(t_span[0], t_span[1] + dt, dt)
        n = len(t_axis)

        # Initialize solution array and set initial condition
        s = np.zeros((n, len(s0), len(s0[0])))
        s[0] = s0
        
        for j in range(n - 1):
            t = t_axis[j]

            s_next = s[j] + dt * f(t, s[j], self.correction, M)
            s_next = s[j] + (dt/2) * (f(t, s[j], self.correction, M) + f(t + dt, s_next, self.correction, M))
            
            s[j+1] = s_next
        return s
    
    def RK3(self):
        """
        
        """
        f = self.slope
        t_span = self.t_span
        dt = self.dt

        s0 = self.two_body_instance.s0
        M = self.two_body_instance.M
        
        t_axis = np.arange(t_span[0], t_span[1] + dt, dt)
        n = len(t_axis)
        
        s = np.zeros((n, len(s0), len(s0[0])))
        s[0] = s0
        
        for j in range(n - 1):
            t = t_axis[j]
                        # RK4 stages
            k1 = f(t, s[j], self.correction, M)
            k2 = f(t + dt/2, s[j] + (dt/2)*k1, self.correction, M)
            k3 = f(t + dt, s[j]-dt*k1 + 2*dt*k2, self.correction, M)
        
            # Combine slopes
            s[j+1] = s[j] + (dt / 6) * (k1 + 4*k2 + k3)

        return s
    
    def scipy_integator(self):
        """
    
        """
        # Fixed method
        method='DOP853'
        t_span = self.t_span
        dt = self.dt    
        slope = self.slope_scipy
        correction = self.correction


        s0 = self.two_body_instance.s0
        M = self.two_body_instance.M

        # Flatten initial condition
        s0_flat = np.concatenate([s0[0], s0[1]])  # [x0, y0, vx0, vy0]

        t_eval = np.arange(t_span[0], t_span[1]+dt, dt)
        
        sol = solve_ivp(slope, args=(correction, M), t_span=(t_eval[0],t_eval[-1]), y0 =s0_flat, method=method, t_eval=t_eval)#, r_tol=1e-7, a_tol=1e-7)
        
        # Reshape solution to (n_steps, 2, 2)
        n_steps = len(sol.t)
        s = np.zeros((n_steps, 2, 2))
        s[:, 0, :] = sol.y[:2, :].T  # Position components
        s[:, 1, :] = sol.y[2:, :].T  # Velocity components

        return s

class RunIntegrator:
    """
    """
    def __init__(self, N, dt, correction, two_body_instance, method, output_dir="."):
        """
        """
        self.N = N # Number of orbits
        self.dt = dt
        self.correction = correction
        self.two_body_instance = two_body_instance
        self.T = self.two_body_instance.T
        self.s0 = self.two_body_instance.s0
        self.t_span = [0, N*self.T]
        self.method = method
        self.output_dir = output_dir

        self.sol = None

    def run(self, save=True):
        """
        
        """
        if self.method == "trapezoidal":
            integrator = Integrators(self.N, self.dt, self.correction, self.two_body_instance)
            self.sol = integrator.trapezoidal()
        elif self.method == "RK3":
            integrator = Integrators(self.N, self.dt, self.correction, self.two_body_instance)
            self.sol = integrator.RK3()
        else:
            integrator = Integrators(self.N, self.dt, self.correction, self.two_body_instance)
            self.sol = integrator.scipy_integator()

        # Unpack the solution
        x = self.sol[:, 0, 0]
        y = self.sol[:, 0, 1]
        vx = self.sol[:, 1, 0]
        vy = self.sol[:, 1, 1]

        t = np.arange(0, self.N * self.T + self.dt, self.dt)
        # Create a DataFrame and save it to a CSV file
        dic = {"t": t, "x": x, "y": y, "vx": vx, "vy": vy}
        df = pd.DataFrame(dic)
        df.to_csv(f"{self.output_dir}/orbit.csv", index=False)
        # Save a plot    
        self.plot_orbit(self.sol, self.s0, self.two_body_instance.a, self.two_body_instance.R_s, self.correction, save)
    
        return self.sol
    
    @staticmethod
    def plot_orbit(sol, s0, a, R_s, correction, save):
        """
        """
        # Unpack initial conditions
        r0 = s0[0]
        
        # Unpack the solution
        x = sol[:, 0, 0]
        y = sol[:, 0, 1]


        # Create plot
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(-2 * a, 2 * a)
        ax.set_ylim(-2 * a, 2 * a)
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title("Two-Body Problem Orbit")
        # Plot central black hole and Schwarzschild radius
        ax.scatter(0, 0, color='k', s=100, label="Black Hole")
        schwarzschild_circle = plt.Circle((0, 0), R_s, color='r', fill=False, linestyle='dashed', label="Schwarzschild Radius")
        ax.add_patch(schwarzschild_circle)
        # Plot initial position of orbiting body
        ax.scatter(r0[0], r0[1], color='b', s=50, label="Planet")
        ax.plot(x, y, color = "orange", label = "orbit")
        ax.legend()
        if save:
            plt.savefig(f"{"relativistic" if correction else "classical"}_orbit.png")

        #plt.show()
        return fig