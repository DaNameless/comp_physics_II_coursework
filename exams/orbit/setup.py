# Import the libraries needed for the set up
from setuptools import find_packages, setup

# To obtain a longer description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(name = "orbit",
      version = "0.1.0",
      author="RSSG", 
      description = "A module containing solvers, animation and plotting methods for the two body problem with a black hole",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/DaNameless/comp_physics_II_coursework/tree/main/exams/orbit",
      license = "BSD",
      packages = find_packages(),
      install_requires = ["numpy==1.26.4", "matplotlib", "scipy", "argparse", "configparser", "pyvista", "scienceplots"],
      python_requires='>=3.9.19'
      )
