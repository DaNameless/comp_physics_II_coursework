# Import the libraries needed for the set up
from setuptools import find_packages, setup

setup(name = "orbit",
      description = "This module contains set of solvers, animation and plotting methods for the two body problem with a black hole",
      author="RSSG", 
      license = "BSD",
      version = "0.1.0",
      packages = find_packages(),
      install_requires = ["numpy", "matplotlib", "scipy", "argparse", "configparser", "pyvista", "scienceplots"])
