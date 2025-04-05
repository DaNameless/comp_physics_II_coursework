# Import the libraries needed for the set up
from setuptools import find_packages, setup

setup(name = "two_body",
      description = "This module contains the two body problem",
      author="RSSG", 
      license = "BSD",
      version = "1.0",
      packages = find_packages(),
      install_requires = ["numpy", "matplotlib", "scipy"])
