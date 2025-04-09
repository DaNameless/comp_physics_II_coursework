# orbit module
This module generates a simulation of the two-body problem with a black hole by solving the **relativistic ODE system** describing the motion of the planet:

$$\frac{d\vec{r}}{dt}=\vec{v}$$

$$m\frac{d\vec{v}}{dt} = -\frac{G\,m\,M}{r^3} \vec{r} \left( 1 + \frac{3\,L^2}{r^2\,c^2} \right)$$

where $L = |\vec{r} \times \vec{v}|$ is the specific angular momentum of the planet, and $c$ is the speed of light. The correction term, $\frac{3\,L^2}{r^2\,c^2}$, accounts for the relativistic precession of the orbit. Note that $m$ cancels out in the above equation.

At $t=0$, we will place the planet at **periapsis** (the closest point in its orbit to the black hole). Thus:

$$x_0 = 0$$

$$y_0 = a\,(1-e)$$

$$v_{x0} = -\sqrt{\frac{G\,M}{a}\frac{1+e}{1-e}}$$

$$v_{y0} = 0$$

where $e$ is the eccentricity of the orbit. You can adjust $e$ to control the orbit shape.

The Schwarzschild radius ($r_s$) of a black hole is the radius of a sphere such that, if all the mass of an object were compressed within that sphere, the escape velocity from the surface of the sphere would equal the speed of light. It is given by:

$$r_s = \frac{2\,G\,M}{c^2}$$

The available methods for solving are: **Trapezoidal**, **RK3**, **Scipy (DOP853)**

## General structure of the module 

The structure of the module `orbit.py` is the following: <br>
```
orbit/                          # Root package directory
│
├── orbit/                      # Main module package
│   ├── __init__.py             # Package initialization
│   └── orbit,py                # Main module/script
│
├── tests/                      # Unit and integration tests
│   └── test_orbit.py           # Unit Testing file
│
├── examples/                   # Example scripts and notebooks
│   ├── config.ini              # Cofig file to initialize script
│   ├── basic_usage.py          # Minimal usage example
│   ├── example_output/
│   │  └
│   └── basic_usage.ipynb       # Advanced example in notebook
│
├── analysis.ipynb              # Analysis Python Notebook
├── analysis_output/
│   └                           # Different vtk files generated for the analysis review
│
├── setup.py                    # Package installation
└── README.md                   # Project overview
```

## Using the module as a script 



### Example:



## Importing the module
You can also import the module in a python notebook for example. 

### Installation


## Author
- [Rolando Sebastián Sánchez García] 