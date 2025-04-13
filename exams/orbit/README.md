# ORBIT
This module generates a simulation of the two-body problem with a black hole by solving the **relativistic ODE system** describing the motion of the planet:

$$\frac{d\vec{r}}{dt}=\vec{v}$$

$$m\frac{d\vec{v}}{dt} = -\frac{G m M}{r^3} \vec{r} \left( 1 + \frac{3 L^2}{r^2 c^2} \right)$$

where $L = \lvert \vec{r} \times \vec{v} \rvert$ is the specific angular momentum of the planet, and $c$ is the speed of light. The correction term, $\frac{3\,L^2}{r^2\,c^2}$, accounts for the relativistic precession of the orbit. Note that $m$ cancels out in the above equation.

At $t=0$, we will place the planet at **periapsis** (the closest point in its orbit to the black hole). Thus:

$$x_0 = 0$$

$$y_0 = a (1-e)$$

$$v_{x0} = -\sqrt{\frac{G M}{a}\frac{1+e}{1-e}}$$

$$v_{y0} = 0$$

where $e$ is the eccentricity of the orbit. You can adjust $e$ to control the orbit shape.

The Schwarzschild radius ($r_s$) of a black hole is the radius of a sphere such that, if all the mass of an object were compressed within that sphere, the escape velocity from the surface of the sphere would equal the speed of light. It is given by:

$$r_s = \frac{2 G M}{c^2}$$

The available methods for solving are: **Trapezoidal**, **RK3**, **Scipy (DOP853)**

---

## General structure of the module 
The structure of the module `orbit.py` is the following: <br>
```
orbit/                          # Root package directory
│
├── orbit/                      # Main module package
│   ├── __init__.py             # Package initialization
│   ├── orbit.py                # Main module/script
│   └── test_orbit.py           # Unit Testing file
│
├── examples/                   # Example scripts and notebooks
│   ├── *.ini                   # Different examples of configuration files
│   ├── example_output/
│   │   ├ *.png                 # Different example plots for the orbits
|   │   ├ *.gif                 # Different example animations for the orbits
|   │   └ *.vtk                 # Different example vtk files for the orbits
│   └── basic_usage.ipynb       # Full example in interactive notebook
│
├── analysis.ipynb              # Analysis Python Notebook
├── outputfolder/
│   └*.vtk                      # Different vtk files generated for the analysis review
│
├── setup.py                    # Package installation
└── README.md                   # Project overview (This file)
```
---

## Usage as script

### Running the Two-Body Problem Simulation

You can run the simulation directly from the command line using:

```bash
python orbit.py --config config.ini
```

This uses a configuration file (`config.ini`) to define the simulation parameters. If you don't provide a config file, the program will fall back to default values.

---

### Available Command Line Options

| Flag               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `-c`, `--config`   | Path to the `.ini` config file (default: `config.ini`)                      |
| `-N`               | Number of orbits to integrate                                               |
| `-a`               | Semi-major axis (in AU)                                                     |
| `-e`               | Orbital eccentricity (0 <= e < 1)                                           |
| `-M`               | Mass of the central body (in Solar masses)                                  |
| `-dt`              | Time step (optional, will be estimated if not provided)                     |
| `-m`               | Integration method: `trapezoidal`, `RK3`, or `scipy` (DOP853)               |
| `-corr`            | Add relativistic correction (flag)                                          |
| `-init_n`          | Name for saving the initial configuration plot                              |
| `-plot_n`          | Name for saving the final orbit plot                                        |
| `-orbit_n`         | Name for saving the orbit VTK file                                          |
| `-anim_n`          | Name for saving the orbit animation (as GIF)                                |
| `-vtk_orbit`       | Path to VTK file used for generating animation                              |
| `-dir`             | Output directory (default: current directory `.`)                           |

---

### Example

Simulate an orbit with relativistic correction using RK3 without a config.ini file:

```bash
python orbit.py -N 3 -a 1.5 -e 0.7 -M 4 -m RK3 -corr \
-init_n rk3_start -plot_n rk3_orbit -orbit_n rk3_orbit \
-anim_n rk3_anim -vtk_orbit rk3_orbit.vtk
```

---

### Output Files

Depending on the options you choose, the following files may be saved in the output directory:

- `init_plot_name.png` → Initial system setup
- `orbit_plot_name.png` → Full orbital trajectory
- `orbit_vtk_name.vtk` → VTK file with orbit data (for animations)
- `animation_name.gif` → Orbit animation with velocity vector and stats


## Importing the module
You can also import the module in a python notebook as show in `examples/basic_usage.ipynb`. 

---
## Installation

To install the `orbit` module and its dependencies, clone the repository and install it using `pip`:

```bash
git clone https://github.com/DaNameless/comp_physics_II_coursework/tree/main/exams/orbit.git
cd orbit
pip install .
```

This will install the package along with all required dependencies:

- `numpy`
- `matplotlib`
- `scipy`
- `pyvista`
- `scienceplots`

> Note: `argparse` and `configparser` are part of Python's standard library, so you don't need to install them separately.
---

### Recommended: Create a Virtual Environment

To avoid conflicts with other Python packages, it’s a good idea to create and activate a virtual environment

## Author
- [Rolando Sebastián Sánchez García] 
