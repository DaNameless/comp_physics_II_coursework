import os
# Modules for parsing the config file and command line arguments
import argparse
import configparser
from pathlib import Path

#Importing necessary libraries for calculations, plotting and writing files
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
import scienceplots # Just for aesthetic purposes
from scipy.integrate import solve_ivp
import pyvista as pv




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
    Class to initialize the two body problem system.
    The system is defined by the mass of the central body, the semi-major axis and the eccentricity.
    The system is defined in a 2D plane, so the initial conditions are defined in the x-y plane.
    It requires numpy and matplotlib to plot the initial conditions.
    
    Author: R.S.S.G.
    Date created: 05/04/2025
    """
    def __init__(self, M, a, e):
        """
        Initialize the two body problem.
        Input:
            M (float) -> Mass of the central body, in our case a black hole, in units of Solar masses.
            a (float)-> Semi-major axis of the orbit, in units of AU. 0<a
            e (float)-> Eccentricity of the orbit. 0<=e<1
          
        Author: R.S.S.G.
        Date created: 05/04/2025
        """        
        # Let's define some fixed variables
        if e >= 1 or e < 0:
            raise ValueError("Eccentricity must be less than 1")
        else:
            self.e = e
            
        if M <= 1e-9:
            raise ValueError("Mass must be greater than 0")
        else:
            self.M = M  

        if a <= 1e-9:
            raise ValueError("Semi-major axis must be greater than 0")
        else:   
            self.a = a

        self.T = 2*np.pi*np.sqrt((a**3)/(G*self.M)) #[years]
        self.R_s = (2*G*self.M)/(C**2)
        self.r0 = np.array([0, a*(1-e)])
        self.v0 = np.array([-np.sqrt((G*self.M/a)*((1+e)/(1-e))),0])
        self.s0 = np.array([self.r0, self.v0])
        
    def plot_grid(self, save=False, output_dir="."):
        """
        This function plots the initial conditions of the two body problem.
        It plots the black hole, the planet and the Schwarzschild radius.
        Input: 
            self
            save (bool) -> If True, saves the plot in the output directory.
            output_dir (str) -> Output directory to save the plot.
        Output:
            orbit_ini.png (png file) -> Plot of the initial setup for the two body problem.
        """
        a = self.a
        e = self.e
        R_s = self.R_s
        r0 = self.r0


        # Create plot
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(-1.4 * a * np.sqrt(1-e**2), 1.4 * a * np.sqrt(1-e**2))
        ax.set_ylim(-1.4 * a * np.sqrt(1-e**2), 1.4 * a * np.sqrt(1-e**2))
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
            plt.savefig(f"{output_dir}/orbit_ini.png")



class Integrators:
    """
    This class implements the integrators for the two body problem.
    It implements the trapezoidal method, the RK3 method and the scipy integrator.
    It also implements the slope function for the two body problem with or without relativistic corrections, depending on user's input.
    It requires numpy and scipy to integrate the equations of motion.

    Author: R.S.S.G.
    Date created: 05/04/2025
    """
    def __init__(self, N, correction, two_body_instance):
        """
        Initialize the integrator.
        Input:
            N (int) -> Number of orbits to integrate.
            correction (bool) -> If True, uses the relativistic correction. False uses the classical two body problem.
            two_body_instance (TwoBodyProblem) -> Instance of the TwoBodyProblem class.
        Author: R.S.S.G.
        Date created: 05/04/2025    
        """
        self.N = N # Number of orbits
        self.correction = correction
        self.two_body_instance = two_body_instance
        self.T = self.two_body_instance.T
        self.t_span = [0, self.N*self.T]
        self.dt = self.t_span[-1]*1e-4 # Time step
        

    @staticmethod
    def slope(t, s, correction, M):
        """
        This function defines the slope of the two body problem ODE.
        Input:
            t (float) -> Time value, however it is not used in the equations of motion.
            s (np.array of float-containing np.arrays) -> State vector (or matrix) [position=[x,y], velocity = [vx,vy]].
            correction (bool) -> If True, uses the relativistic correction. False uses the classical two body problem.
            M (float) -> Mass of the central body, in our case a black hole, in units of Solar masses.
        Author: R.S.S.G.
        Date created: 05/04/2025  
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
        This function defines the slope of the two body problem ODE. This version is used for the scipy integrator.
        It flattens the state vector to be compatible with scipy's solve_ivp.
        It also uses the same equations as the alternative slope function, but it is more efficient for scipy's integrator.
        Input:
            t (float) -> Time value, however it is not used in the equations of motion.
            s (np.array of float-containing np.arrays) -> State vector (or matrix) [position=[x,y], velocity = [vx,vy]].
            correction (bool) -> If True, uses the relativistic correction. False uses the classical two body problem.
            M (float) -> Mass of the central body, in our case a black hole, in units of Solar masses.
        Author: R.S.S.G.
        Date created: 05/04/2025  
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
        This function implements the trapezoidal method for the two body problem.
        It uses the `slope` static method defined to compute the derivatives.
        It stops when the solution diverges, using the difference with the consecutive to be less than 1e-5.
        Input:
            self
        Output:
            s (np.array of float-containing np.arrays) -> State vector (or matrix) [position=[x,y], velocity = [vx,vy]].
            t (np.array of float) -> Time vector axis.

        Author: R.S.S.G.
        Date created: 05/04/2025  
        """
        f = self.slope
        t_span = self.t_span
        dt = self.dt
        
        s0 = self.two_body_instance.s0
        M = self.two_body_instance.M

        # Initialize lists to store results dynamically
        s_list = [s0.copy()]
        t_list = [t_span[0]]

        # Maximum allowed value before we consider it diverged
        MAX_VALUE = 1e5

        for j in range(1, int((t_span[1] - t_span[0]) / dt) + 1):
            t_prev = t_list[-1]
            s_prev = s_list[-1]
            
            try:
                # Predictor step
                s_predict = s_prev + dt * f(t_prev, s_prev, self.correction, M)
                
                # Corrector step (trapezoidal rule)
                s_next = s_prev + (dt/2) * (
                    f(t_prev, s_prev, self.correction, M) + 
                    f(t_prev + dt, s_predict, self.correction, M)
                )
                
                # Check for divergence
                if (np.any(np.isnan(s_next)) or (np.any(np.abs(s_next-s_predict) > MAX_VALUE))):
                    break
                    
                # Store results
                s_list.append(s_next)
                t_list.append(t_prev + dt)
                
            except (FloatingPointError, ValueError):
                # Handle numerical errors (overflow, etc.)
                break

        # Convert lists to numpy arrays
        s = np.array(s_list)
        t_array = np.array(t_list)

        return s, t_array
    
    def RK3(self):
        """
        This function implements the RK3 method for the two body problem.
        It uses the `slope` static method defined to compute the derivatives.
        It stops when the solution diverges, using the difference with the consecutive to be less than 1e-5.
        Input:
            self
        Output:
            s (np.array of float-containing np.arrays) -> State vector (or matrix) [position=[x,y], velocity = [vx,vy]].
            t (np.array of float) -> Time vector axis.
        
        Author: R.S.S.G.
        Date created: 05/04/2025 
        """
        f = self.slope
        t_span = self.t_span
        dt = self.dt

        s0 = self.two_body_instance.s0
        M = self.two_body_instance.M

        # Initialize lists to store results dynamically
        s_list = [s0.copy()]
        t_list = [t_span[0]]

        # Maximum allowed value before we consider it diverged
        MAX_VALUE = 1e5 

        for j in range(1, int((t_span[1] - t_span[0]) / dt) + 1):
            t_prev = t_list[-1]
            s_prev = s_list[-1]
            
            # RK4 stages
            try:
                k1 = f(t_prev, s_prev, self.correction, M)
                k2 = f(t_prev + dt/2, s_prev + (dt/2)*k1, self.correction, M)
                k3 = f(t_prev + dt, s_prev - dt*k1 + 2*dt*k2, self.correction, M)
                
                # Compute next step
                s_next = s_prev + (dt / 6) * (k1 + 4*k2 + k3)
                
                # Check for divergence
                if (np.any(np.isnan(s_next)) or (np.any(np.abs(s_next-s_prev) > MAX_VALUE))):
                    break
                    
                # Store results
                s_list.append(s_next)
                t_list.append(t_prev + dt)
            except (FloatingPointError, ValueError):
                # Handle numerical errors (overflow, etc.)
                break

        # Convert lists to numpy arrays
        s = np.array(s_list)
        t_array = np.array(t_list)

        return s, t_array
    
    def scipy_integator(self):
        """
        This function implements the scipy solve_ivp integrator for the two body problem.
        It uses the `slope_scipy` static method defined to compute the derivatives.
        It stops when the solution diverges, but this is handled by the scipy integrator.
        Input:
            self
        Output:
            s (np.array of float-containing np.arrays) -> State vector (or matrix) [position=[x,y], velocity = [vx,vy]].
            sol.t (np.array of float) -> Time vector axis.
        Author: R.S.S.G.
        Date created: 05/04/2025 
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
        
        sol = solve_ivp(slope, args=(correction, M), t_span=(t_eval[0],t_eval[-1]), y0 =s0_flat, method=method, t_eval=t_eval)# r_tol = 1e-8 ,a_tol=1e-8)
        # Reshape solution to (n_steps, 2, 2)
        n_steps = len(sol.t)
        s = np.zeros((n_steps, 2, 2))
        s[:, 0, :] = sol.y[:2, :].T  # Position components
        s[:, 1, :] = sol.y[2:, :].T  # Velocity components

        return s, sol.t

class RunIntegrator:
    """
    This class runs the integrator for the two body problem by using the `Integrators` class to run the integrator.
    It also implements a plot function to plot the complete planet's orbit.
    It requires numpy, matplotlib and pyvista to write and read VTK files.
    Author: R.S.S.G.
    Date created: 05/04/2025 
    """
    def __init__(self, N, correction, two_body_instance, method, output_dir, save):
        """
        Initialize the integrator. Automatically sets the time span and the initial conditions.
        It also sets the output directory and the save option.
        It uses the `Integrators` class to run the integrator.
        Input:
            N (int) -> Number of orbits to integrate.
            correction (bool) -> If True, uses the relativistic correction. False uses the classical two body problem.
            two_body_instance (TwoBodyProblem) -> Instance of the TwoBodyProblem class.
            method (str) -> Integration method to use. Options are "trapezoidal", "RK3" or "scipy". 
            output_dir (str) -> Output directory to save the plot.
            save (bool) -> If True, saves the plot in the output directory.

        Author: R.S.S.G.
        Date created: 05/04/2025 
        """
        self.N = N # Number of orbits
        self.correction = correction
        self.two_body_instance = two_body_instance
        self.T = self.two_body_instance.T
        self.s0 = self.two_body_instance.s0
        self.a = self.two_body_instance.a
        self.R_s = self.two_body_instance.R_s
        self.e = self.two_body_instance.e
        self.t_span = [0, self.N*self.T]
        if method not in ["trapezoidal", "RK3", "scipy"]:
            raise ValueError("Invalid integration method")
        else: 
            self.method = method

        self.output_dir = output_dir
        self.save = save

        self.sol = None

    def run(self):
        """
        This function runs the integrator for the two body problem and plots the orbit using the static method `plot_orbit`.
        It uses the `Integrators` class to run the integrator.
        It saves the orbit as a VTK file using `pyvista`.

        Input:
            self
        Output:
            sol (np.array of float-containing np.arrays) -> State vector (or matrix) [position=[x,y], velocity = [vx,vy]].
            orbit.vtk (vtk file) -> VTK file with the orbit data.
        Author: R.S.S.G.
        Date created: 05/04/2025 
        """
        if self.method == "trapezoidal":
            integrator = Integrators(self.N, self.correction, self.two_body_instance)
            self.sol = integrator.trapezoidal()
        elif self.method == "RK3":
            integrator = Integrators(self.N, self.correction, self.two_body_instance)
            self.sol = integrator.RK3()
        elif self.method == "scipy":
            integrator = Integrators(self.N, self.correction, self.two_body_instance)
            self.sol = integrator.scipy_integator()
        else:
            raise ValueError("Invalid integration method")

        # Unpack and round to 5 decimals
        x = np.around(self.sol[0][:, 0, 0], decimals=5)
        y = np.around(self.sol[0][:, 0, 1], decimals=5)
        z = np.zeros_like(x)  # 3D coordinates
        vx = np.around(self.sol[0][:, 1, 0], decimals=5)
        vy = np.around(self.sol[0][:, 1, 1], decimals=5)
        vz = np.zeros_like(vx)
        t_eval = np.around(self.sol[1], decimals=5)

        points = np.column_stack([x, y, z])
        orbit = pv.PolyData(points)
        
        # Add velocity vectors
        orbit['velocity'] = np.column_stack([vx, vy, vz])
        orbit['time'] = t_eval
        
        # Add metadata as field data
        orbit.field_data['a'] = [np.around(self.a, 5)]
        orbit.field_data['e'] = [np.around(self.e, 5)]
        orbit.field_data['n_orbits'] = [self.N]
        orbit.field_data['schwarzschild_radius'] = [np.around(self.R_s, 5)]
        orbit.field_data['correction_enabled'] = [1 if self.correction else 0]

        # Save as VTK file
        vtk_filename = f"{self.output_dir}/orbit.vtk"
        orbit.save(vtk_filename)

        # Save a plot    
        self.plot_orbit(self.sol[0], self.s0, self.a, self.e, self.R_s, self.correction, self.save, self.output_dir)
    
        return self.sol
    
    @staticmethod
    def plot_orbit(sol, s0, a, e, R_s, correction, save, output_dir):
        """
        This function plots the orbit of the two body problem.
        It uses the `matplotlib` library to plot the orbit.

        Input:
            sol (np.array of float-containing np.arrays) -> State vector (or matrix) [position=[x,y], velocity = [vx,vy]].
            s0 (np.array of float-containing np.arrays) -> Initial conditions [position=[x,y], velocity = [vx,vy]].
            a (float) -> Semi-major axis of the orbit, in units of AU.
            e (float) -> Eccentricity of the orbit. 0<=e<1
            R_s (float) -> Schwarzschild radius of the black hole, in units of AU.
            correction (bool) -> If True, uses the relativistic correction. False uses the classical two body problem.
            save (bool) -> If True, saves the plot in the output directory.
            output_dir (str) -> Output directory to save the plot.
        Output:
            orbit.png (png file) -> Plot of the planet's orbit.

        Author: R.S.S.G.
        Date created: 05/04/2025 

        """
        # Unpack initial conditions
        r0 = s0[0]
        
        # Unpack the solution
        x = sol[:, 0, 0]
        y = sol[:, 0, 1]


        # Create plot
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(-1.6 * a * np.sqrt(1 - e**2), 1.6 * a * np.sqrt(1 - e**2))
        
        # To avoid the orbit to be cut in half, we need to set the limits
        # to the maximum and minimum values of the orbit
        #ax.set_ylim(-1.2 * self.a * (1 + self.e), 1.2 * self.a * (1 - self.e))   # This would work if the orbit is classical
        ax.set_ylim(1.2 * min(y) , 1.4 * a * (1 - e)) # This will work for both classical and relativistic orbits
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
            orbit_type = "relativistic" if correction else "classical"
            plt.savefig(f"{output_dir}/{orbit_type}_orbit.png")

        plt.close()
        return fig
    
class Animation:
    """ 
    Class to animate the two body problem orbit.
    It uses the `matplotlib` library to create the animation. It requires the `pyvista` library to read the VTK file and extract the orbit data.
    It also uses the `numpy` library to manipulate the data.
    It uses the `matplotlib.animation` library to create the animation.
    It requires the `os` library to create the output directory if it does not exist.

    Author: R.S.S.G.
    Date created: 05/04/2025 
    """
    def __init__(self, orbit_file_dir, save_dir=None, fps=30):
        """
        Initialize the animation class. It reads the VTK file and extracts the orbit data.
        It also sets the output directory and the frames per second for the animation.
        It uses the `pyvista` library to read the VTK file and extract the orbit data.
        Input:
            orbit_file_dir (str) -> Path to the VTK file with the orbit data.
            save_dir (str) -> Output directory to save the animation. If None, it will not save the animation.
            fps (int) -> Frames per second for the animation. Default is 30.
        
        Author: R.S.S.G.
        Date created: 05/04/2025 
        """
        self.orbit_file_dir = orbit_file_dir
        self.save_dir = save_dir
        self.fps = fps
        self.orbit = pv.read(orbit_file_dir)

        # Extract positions (x, y, z)
        points = self.orbit.points  # shape (N, 3)
        self.x = points[:, 0]       # x positions
        self.y = points[:, 1]       # y positions
        z = points[:, 2]       # z positions (zeros as our orbit 2D)

        # Extract velocities (vx, vy, vz)
        velocity = self.orbit['velocity']  # shape (N, 3)
        self.vx = velocity[:, 0]          # vx components
        self.vy = velocity[:, 1]          # vy components
        vz = velocity[:, 2]          # vz components (zeros as our orbit 2D)

        # Extract time values
        self.t = self.orbit['time'] 

        # Extract orbital parameters (metadata)
        self.a = self.orbit.field_data['a'][0]
        self.e = self.orbit.field_data['e'][0]
        self.N = self.orbit.field_data['n_orbits'][0]
        self.R_s = self.orbit.field_data['schwarzschild_radius'][0]
        self.correction = bool(self.orbit.field_data['correction_enabled'][0])

    def animate(self):
        """
        This function creates the animation of the two body problem orbit. It implements the orbit as a gif file, with velocity vector pointing in the direction of the movement.
        It uses the `matplotlib.animation` library to create the animation.
        It requires the `os` library to create the output directory if it does not exist.
        Input:
            self
        Output:
            anim (matplotlib.animation.FuncAnimation) -> Animation object.
            orbit.gif (gif file) -> Animation of the two body problem orbit.
        Author: R.S.S.G.
        Date created: 05/04/2025 

        """
        save_dir = self.save_dir

        # Create plot
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim(-1.7 * self.a * np.sqrt(1 - self.e**2), 1.7 * self.a * np.sqrt(1 - self.e**2))
        
        # To avoid the orbit to be cut in half, we need to set the limits
        # to the maximum and minimum values of the orbit
        #ax.set_ylim(-1.2 * self.a * (1 + self.e), 1.2 * self.a * (1 - self.e))   # This would work if the orbit is classical
        ax.set_ylim(1.2 * min(self.y) , 1.4 * self.a * (1 - self.e)) # This will work for both classical and relativistic orbits
        
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title("Two-Body Problem Orbit")

        # Central body and Schwarzschild radius
        ax.scatter(0, 0, color='k', s=100, label="Black Hole")
        schwarzschild_circle = plt.Circle((0, 0), self.R_s, color='r', fill=False, linestyle='dashed', label="Schwarzschild Radius")
        ax.add_patch(schwarzschild_circle)

        # Full orbit path (static background)
        ax.plot(self.x, self.y, color="orange", label="Orbit", linestyle='--', alpha=0.5)

        # Dynamic elements
        point, = ax.plot([], [], 'bo', markersize=10, label="Planet", zorder=10)
        trail_line, = ax.plot([], [], color='blue', linewidth=1.5, alpha=0.7, label="Trajectory", zorder=5)

        # Velocity vector setup
        max_vel = max(np.max(np.abs(self.vx)), np.max(np.abs(self.vy)))
        arrow_scale = 0.2 * self.a / max_vel  # Smaller scale for clarity

        velocity_arrow = ax.quiver(0, 0, 0, 0,\
                                color='dodgerblue',\
                                angles='xy',\
                                scale_units='xy',\
                                scale=1,\
                                width=0.008,\
                                headwidth=4,\
                                headlength=5,\
                                headaxislength=4.5,\
                                zorder=9)

        # Dynamic text
        legend_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9, linespacing=1.5, verticalalignment='top', horizontalalignment='left',\
                                bbox=dict(boxstyle='round,pad=0.4',\
                                    facecolor='whitesmoke',\
                                    edgecolor='gray',\
                                    linewidth=1,\
                                    alpha=0.95\
                                )
                            )

        ax.legend()

        # Adaptive sampling
        total_frames = len(self.t)
        target_frames = 100
        if total_frames <= target_frames:
            step = 1
            sampled_indices = range(total_frames)
        else:
            step = max(1, total_frames // target_frames)
            sampled_indices = range(0, total_frames, step)

        # Frame update function
        def animate_frame(i):
            x_i = self.x[i]
            y_i = self.y[i]
            vx_i = self.vx[i]
            vy_i = self.vy[i]
            v_mod_i = np.sqrt(vx_i**2 + vy_i**2)

            # Update planet and trail
            point.set_data([x_i], [y_i])
            trail_line.set_data(self.x[:i+1], self.y[:i+1])

            # Update velocity vector (arrow from current position)
            velocity_arrow.set_offsets([x_i, y_i])
            velocity_arrow.set_UVC(vx_i * arrow_scale, vy_i * arrow_scale)

            # Update text
            legend_text.set_text(
                f"Current Position [AU]:\n"
                f"(x = {x_i:.3f}, y = {y_i:.3f})\n"
                f"Current Velocity [AU/yr]: \n v = {v_mod_i:.3e}\n"
                f"(vx = {vx_i:.3e}, vy = {vy_i:.3e})"
            )

            return point, trail_line, velocity_arrow, legend_text

        # Animate
        anim = animation.FuncAnimation(fig, animate_frame, frames=sampled_indices, interval=50, blit=True, repeat=True)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            gif_output = os.path.join(save_dir, "orbit.gif")
            anim.save(gif_output, writer="pillow", fps=20, dpi=100)

        plt.close()

        return anim
        


def parse_config_file(config_path):
    """
    Parse the configuration file (.ini) and return the default values.
    Input:
        config_path (str) -> Path to the configuration file (.ini).
    Output:
        defaults (dict) -> Dictionary with the default values.
    Author: R.S.S.G.
    Date created: 05/04/2025 
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    defaults = {
        'N': config.getint('two_body', 'N', fallback=1),
        'a': config.getfloat('two_body', 'a', fallback=1.0),
        'e': config.getfloat('two_body', 'e', fallback=0.0),
        'M': config.getfloat('two_body', 'M', fallback=1.0),
        'method': config.get('two_body', 'method', fallback='scipy'),
        'correction': config.getboolean('two_body', 'correction', fallback=False),
        'save_init_plot': config.getboolean('two_body', 'save_init_plot', fallback=False),
        'save_plot': config.getboolean('two_body', 'save_plot', fallback=False),
        'output_dir': config.get('two_body', 'output_dir', fallback='.'),
        'animate': config.getboolean('two_body', 'animate', fallback=False),
    }
    return defaults


def main():
    """
    Main function to run the two body problem solver.
    It parses the command line arguments and the configuration file.
    It creates the output directory if it does not exist.
    It runs the integrator and saves the desired results.
    It creates the animation if requested.
    Input for parser:
        -c, --config (str) -> Path to the configuration file (.ini).
        -N, --N (int) -> Number of orbits to integrate.
        -a, --a (float) -> Semi-major axis of the orbit, in units of AU.    
        -e, --e (float) -> Eccentricity of the orbit. 0<=e<1
        -M, --M (float) -> Mass of the central body, in units of Solar masses.
        -m, --method (str) -> Integration method to use. Options are "trapezoidal", "RK3" or "scipy".
        -corr, --correction (bool) -> If True, uses the relativistic correction. False uses the classical two body problem.
        -save_init, --save_init_plot (bool) -> If True, saves the initial setup plot.
        -save_plot, --save_plot (bool) -> If True, saves the orbit plot.
        -dir, --output_dir (str) -> Output directory to save the plot.
        -anim, --animate (bool) -> If True, creates the animation.
    Output:
        Requested plots and VTK files with the orbit data.
    
    Author: R.S.S.G.
    Date created: 05/04/2025 
    """
    # Parse config file if it exists
    config_path = Path('config.ini')
    defaults = parse_config_file(config_path) if config_path.exists() else {}

    # Parse command line arguments (which will override config file)
    parser = argparse.ArgumentParser(description="Two Body Problem Solver")
    parser.add_argument("-c", "--config", type=str, default="config.ini", help="Path to config file")
    parser.add_argument("-N", "--N", type=int, default=defaults.get('N', 1), help="Number of orbits")
    parser.add_argument("-a", "--a", type=float, default=defaults.get('a', 1), help="Semi-major axis")
    parser.add_argument("-e", "--e", type=float, default=defaults.get('e', 0), help="Eccentricity")
    parser.add_argument("-M", "--M", type=float, default=defaults.get('M', 1), help="Mass of the Black Hole")
    parser.add_argument("-m", "--method", type=str, default=defaults.get('method', 'scipy'), help="Integration method")
    parser.add_argument("-corr", "--correction", action='store_true', default=defaults.get('correction', False), 
                       help="Use relativistic correction")
    parser.add_argument("-save_init", "--save_init_plot", action='store_true', 
                       default=defaults.get('save_init_plot', False), help="Save the initial setup plot")
    parser.add_argument("-save_plot", "--save_plot", action='store_true', 
                       default=defaults.get('save_plot', False), help="Save the orbit plot")
    parser.add_argument("-dir", "--output_dir", type=str, default=defaults.get('output_dir', '.'), 
                       help="Output directory for results")
    parser.add_argument("-anim", "--animate", action='store_true', default=defaults.get('animate', False), 
                       help="Create animation")
    
    args = parser.parse_args()
    
    # If a different config file was specified, use it
    if args.config != 'config.ini' or not config_path.exists():
        defaults = parse_config_file(args.config)
        # Update args with values from the specified config file
        for key, value in defaults.items():
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
    
    N = args.N
    a = args.a
    e = args.e
    M = args.M
    method = args.method
    correction = args.correction
    save_init = args.save_init_plot
    save = args.save_plot
    output_dir = args.output_dir
    animate = args.animate
    
    two_body_instance = TwoBodyProblem(M, a, e)
    two_body_instance.plot_grid(save_init, output_dir)
    run_integrator = RunIntegrator(N, correction, two_body_instance, method, output_dir, save)
    sol = run_integrator.run()
    
    if animate:
        orbit_file_dir = f"{output_dir}/orbit.vtk"
        animation_instance = Animation(orbit_file_dir, output_dir)
        animation_instance.animate()


if __name__ == "__main__":
    # Run the main function
    main()