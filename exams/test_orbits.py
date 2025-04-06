import pytest
import numpy as np
import os
import shutil
from two_body_problem import TwoBodyProblem, RunIntegrator  # Replace with your actual module

# --------------------------
# Test Class with Setup/Teardown
# --------------------------
class TestOrbitSimulation:
    """Test class with setup/teardown methods"""
    
    @classmethod
    def setup_class(cls):
        """Class-level setup - runs once before all tests"""
        cls.test_output_dir = "test_outputs"
        os.makedirs(cls.test_output_dir, exist_ok=True)
        cls.base_params = {
            "M": 1.0,
            "a": 1.0,
            "e": 0.1,
            "N": 1,
            "method": "RK3",
            "correction": False,
            "save": False,
        }
        print("\n=== Test Suite Starting ===")

    @classmethod
    def teardown_class(cls):
        """Class-level teardown - runs once after all tests"""
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
        print("\n=== Test Suite Complete ===")

    def setup_method(self, method):
        """Method-level setup - runs before each test"""
        self.two_body = TwoBodyProblem(
            M=self.base_params["M"],
            a=self.base_params["a"],
            e=self.base_params["e"]
        )
        print(f"\nSetting up for {method.__name__}")

    def teardown_method(self, method):
        """Method-level teardown - runs after each test"""
        del self.two_body
        print(f"Tearing down after {method.__name__}")

    # --------------------------
    # Test Cases
    # --------------------------

    def test_valid_input_handling(self):
        """Verify all input parameters meet physical constraints"""
        # Test base parameters are valid
        assert isinstance(self.base_params["M"], (int, float)), "Mass must be numeric"
        assert self.base_params["M"] > 0, "Mass must be positive"
        
        assert isinstance(self.base_params["a"], (int, float)), "Semi-major axis must be numeric"
        assert self.base_params["a"] > 0, "Semi-major axis must be positive"
        
        assert isinstance(self.base_params["e"], (int, float)), "Eccentricity must be numeric"
        assert 0 <= self.base_params["e"] < 1, "Eccentricity must be 0 ≤ e < 1"
        
        assert isinstance(self.base_params["N"], int), "Orbit count must be integer"
        assert self.base_params["N"] > 0, "Orbit count must be positive"
        
        assert self.base_params["method"] in ["RK3", "scipy", "trapezoidal"], "Invalid method"
        assert isinstance(self.base_params["correction"], bool), "Relativistic flag must be boolean"
        assert isinstance(self.base_params.get("save", False), bool), "Save flag must be boolean"

        # Verify TwoBodyProblem initialization enforces constraints
        with pytest.raises(ValueError):
            TwoBodyProblem(M=0, a=1.0, e=0.1)  # Invalid mass
        
        with pytest.raises(ValueError):
            TwoBodyProblem(M=1.0, a=-1.0, e=0.1)  # Invalid semi-major axis
        
        with pytest.raises(ValueError):
            TwoBodyProblem(M=1.0, a=1.0, e=1.1)  # Invalid eccentricity

        # Test RunIntegrator with valid inputs
        integrator = RunIntegrator(
            N=self.base_params["N"],
            correction=self.base_params["correction"],
            two_body_instance=self.two_body,
            method=self.base_params["method"],
            output_dir=self.test_output_dir,
            save=self.base_params.get("save", False)
        )
        
        sol, t = integrator.run()
        
        # Verify solution properties
        assert len(sol) == len(t), "Solution/time arrays should match length"
        assert not np.any(np.isnan(sol)), "Solution should not contain NaNs"
        assert np.all(np.isfinite(sol)), "Solution should be finite"



    def test_invalid_method_handling(self):
        """Verify invalid method rejection"""
        with pytest.raises(ValueError, match="Invalid integration method"):
            RunIntegrator(
                N=1,
                correction=False,
                two_body_instance=self.two_body,
                method="INVALID_METHOD",
                output_dir=self.test_output_dir,
                save=self.base_params["save"],
            )

    @pytest.mark.parametrize("param,value,expected_change", [
        ("N", 3, "orbit_count"),
        ("e", 0.9, "eccentricity"),
        ("M", 10.0, "central_mass")
    ])
    def test_input_sensitivity(self, param, value, expected_change):
        """Verify input changes produce different outputs"""
        # Create modified parameters
        modified_params = self.base_params.copy()
        modified_params[param] = value
        
        # Baseline solution
        base_integrator = RunIntegrator(
            N=self.base_params["N"],
            correction=self.base_params["correction"],
            two_body_instance=self.two_body,
            method=self.base_params["method"],
            output_dir=self.test_output_dir,
            save=self.base_params["save"],
        )
        base_sol, _ = base_integrator.run()
        
        # Modified solution
        mod_body = TwoBodyProblem(
            M=modified_params["M"],
            a=modified_params["a"],
            e=modified_params["e"]
        )
        mod_integrator = RunIntegrator(
            N=modified_params["N"],
            correction=modified_params["correction"],
            two_body_instance=mod_body,
            method=modified_params["method"],
            output_dir=self.test_output_dir,
            save=self.base_params["save"],
        )
        mod_sol, _ = mod_integrator.run()
        
        # Assign the same length to both solutions
        min_length = min(len(base_sol), len(mod_sol))
        base_sol = base_sol[:min_length]
        mod_sol = mod_sol[:min_length]

        assert not np.allclose(base_sol, mod_sol), \
            f"Changing {expected_change} should produce different orbits"
