import pytest
import numpy as np
import os
import shutil # We will use it to recursively delete entire directory trees
from orbit import TwoBodyProblem, RunIntegrator



class TestOrbitSimulation:
    """
    Test class with setup/teardown methods to test on correct input values from the user, 
    handling invalid input methods, and uniqueness of solutions.
    
    This class tests the TwoBodyProblem and RunIntegrator classes from the orbit module,
    verifying their functionality under various conditions.
    
    Author: R.S.S.G.
    Date created: 05/04/2025
    """
    
    @classmethod
    def setup_class(cls):
        """
        Class-level setup - runs once before all tests.
        Creates test output directory and defines some base simulation parameters.
        
        Author: R.S.S.G.
        Date created: 05/04/2025
        """
        cls.test_output_dir = "test_outputs"  # Directory for test outputs
        os.makedirs(cls.test_output_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        # Base parameters for all tests (these will be overridden if needed in the individual tests)
        cls.base_params = {
            "M": 1.0,          # Black hole mass (in solar masses)
            "a": 1.0,          # Semi-major axis (AU)
            "e": 0.1,          # Eccentricity (0 <= e < 1)
            "N": 1,            # Number of periods 
            "dt": 0.001,             # Time_step
            "method": "scipy", # Integration method
            "correction": True,      # Relativistic correction (INI: True)
            "vtk_name": "test_orbit", # vtk_orbit name to be saved and used
            "orbit_plot_name": "test_orbit_plot",
            "output_dir": "./test_outputs"  # Explicit output directory
        }
        print("\n=== Test Suite Starting ===")

    @classmethod
    def teardown_class(cls):
        """
        Class-level teardown - runs once after all tests.
        Cleans up by removing the test output directory.
        
        Author: R.S.S.G.
        Date created: 05/04/2025
        """
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)  # Remove directory and all contents
        print("\n=== Test Suite Complete ===")

    def setup_method(self, method):
        """
        Method-level setup - runs before each test method.
        Creates a TwoBodyProblem instance with the base parameters previously defined and uses a given method.
        
        Author: R.S.S.G.
        Date created: 05/04/2025
        """
        self.two_body = TwoBodyProblem(
            M=self.base_params["M"],  # Initialize with base mass
            a=self.base_params["a"],  # Initialize with base semi-major axis
            e=self.base_params["e"]   # Initialize with base eccentricity
        )
        print(f"\nSetting up for {method.__name__}")

    def teardown_method(self, method):
        """
        Method-level teardown - runs after each test method.
        Cleans up the test instance.
        
        Author: R.S.S.G.
        Date created: 05/04/2025
        """
        del self.two_body  # Delete the TwoBodyProblem instance
        print(f"Tearing down after {method.__name__}")

    
    # Unit Test Cases
    def test_valid_input(self):
        """
        Test that input parameters meet physical constraints and are properly validated.

        Author: R.S.S.G.
        Date created: 05/04/2025
        """

        # Let's check for the type and if they are physically accepted values
        assert isinstance(self.base_params["M"], (int, float)), "Mass must be a real number"
        assert self.base_params["M"] >= 1, "Mass must be positive and at least one solar mass"
        
        assert isinstance(self.base_params["a"], (int, float)), "Semi-major axis must be a real number"
        assert self.base_params["a"] > 0, "Semi-major axis must be positive"
        
        assert isinstance(self.base_params["e"], (int, float)), "Eccentricity must be a real number"
        assert 0 <= self.base_params["e"] < 1, "Eccentricity must be 0 <= e < 1"
        
        assert isinstance(self.base_params["N"], int), "Number of orbit periods must be an integer number"
        assert self.base_params["N"] > 0, "Number of orbit periods must be positive"
        
        assert self.base_params["method"] in ["RK3", "scipy", "trapezoidal"], "Invalid method"

        assert isinstance(self.base_params["correction"], bool), "Relativistic flag must be boolean"
    
        assert isinstance(self.base_params["dt"], (int, float)), "Timestep must be a real number if given, otherwise should be None"
        assert self.base_params["dt"] > 0, "Timestep must be positive"
        
        assert self.base_params["method"] in ["RK3", "scipy", "trapezoidal"], "Invalid method"
        
        assert self.base_params["correction"] is True, "Should enable relativistic correction"

        with pytest.raises(ValueError):
            TwoBodyProblem(M=0, a=1.0, e=0.1)  # Invalid mass (0)
        
        with pytest.raises(ValueError):
            TwoBodyProblem(M=1.0, a=-1.0, e=0.1)  # Invalid semi-major axis (negative)
        
        with pytest.raises(ValueError):
            TwoBodyProblem(M=1.0, a=1.0, e=1.1)  # Invalid eccentricity (≥1)

        # Test RunIntegrator with valid inputs
        integrator = RunIntegrator(
            N=self.base_params["N"],              # Number of orbits
            correction=self.base_params["correction"],  # Relativistic correction
            dt = self.base_params["dt"],  # Time_step 
            two_body_instance=self.two_body,        # TwoBodyProblem instance
            method=self.base_params["method"],      # Integration method
            output_dir=self.test_output_dir,       # Output directory
            vtk_name=self.base_params["vtk_name"],  # Saving vtk name
            orbit_plot_name=self.base_params["orbit_plot_name"]  # Saving plot name
        )
        
        # Run the integrator and get solution
        sol, t = integrator.run()
        
        # Verify solution properties
        assert len(sol) == len(t), "Solution and time arrays should have matching lengths"
        assert not np.any(np.isnan(sol)), "Solution should not contain NaN values"
        assert np.all(np.isfinite(sol)), "Solution should contain only finite values"

    def test_invalid_method(self):
        """
        Test that invalid integration methods are properly rejected.
        Verifies the system raises ValueError for unsupported methods.
        
        Author: R.S.S.G.
        Date created: 05/04/2025      
        """
        
        with pytest.raises(ValueError, match="Invalid integration method"):
            RunIntegrator(
                N=1,  # Number of orbits
                correction=False,  # No relativistic correction
                dt = self.base_params["dt"],  # Time_step 
                two_body_instance=self.two_body,  # TwoBodyProblem instance
                method="INVALID_METHOD",  # This should be invalid
                output_dir=self.test_output_dir,       # Output directory
                vtk_name=self.base_params["vtk_name"],  # Saving vtk name
                orbit_plot_name=self.base_params["orbit_plot_name"]  # Saving plot name
            )

    @pytest.mark.parametrize("param,value,expected_change", [
        ("N", 3, "number_of_orbits"),      # Test changing number of orbits
        ("e", 0.9, "eccentricity"),   # Test changing eccentricity
        ("M", 10.0, "black_hole_mass") , # Test changing black hole's mass
        ("a", 2.0, "semi_major_axis") # Test changing semi-major axis length  
    ])
    def test_uniqueness(self, param, value, expected_change):
        """
        Test that different parameters give different orbit solutions.
        Uses parameterized testing to efficiently test multiple scenarios.
        
        Input:
            param -> The parameter to be modified (N, e, M, a)
            value (float) -> The new value that the parameter will get to be tested
            expected_change (str) -> Description of what's being changed to be reported in the error messages
        
        Author: R.S.S.G.
        Date created: 05/04/2025   
        """
        
        # Create modified parameters by copying base and updating one parameter
        modified_params = self.base_params.copy()
        modified_params[param] = value
        
        # Baseline solution with original parameters
        base_integrator = RunIntegrator(
            N=self.base_params["N"],
            correction=self.base_params["correction"],
            dt = self.base_params["dt"],  # Time_step 
            two_body_instance=self.two_body,
            method=self.base_params["method"],
            output_dir=self.test_output_dir,       # Output directory
            vtk_name=self.base_params["vtk_name"],  # Saving vtk name
            orbit_plot_name=self.base_params["orbit_plot_name"]  # Saving plot name
        )
        base_sol, _ = base_integrator.run()
        
        # Modified solution with changed parameter
        mod_body = TwoBodyProblem(
            M=modified_params["M"],
            a=modified_params["a"],
            e=modified_params["e"],
        )

        mod_integrator = RunIntegrator(
            N=modified_params["N"],
            correction=modified_params["correction"],
            dt = self.base_params["dt"],  # Time_step 
            two_body_instance=mod_body,
            method=modified_params["method"],
            output_dir=self.test_output_dir,       # Output directory
            vtk_name=self.base_params["vtk_name"],  # Saving vtk name
            orbit_plot_name=self.base_params["orbit_plot_name"]  # Saving plot name
        )
        mod_sol, _ = mod_integrator.run()
        
        # Ensure we compare arrays of the same length
        min_length = min(len(base_sol), len(mod_sol))
        base_sol = base_sol[:min_length]
        mod_sol = mod_sol[:min_length]

        # Verify solutions are different
        assert not np.allclose(base_sol, mod_sol), \
            f"Changing {expected_change} should produce different orbital solutions"
