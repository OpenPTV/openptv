import numpy as np
from .ray_tracing import ray_tracing

def test_ray_tracing():
    # Test Case 1
    x, y = 0, 0
    cal = # create Variable for cal with necessary data 
    mm = # create an object for mm with necessary data
    expected_output = np.array([cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0]), np.zeros(3)
    assert np.allclose(ray_tracing(x, y, cal, mm), expected_output)
    
    # Test Case 2
    x, y = 1, 2
    cal = # create Variable for cal with necessary data 
    mm = # create an object for mm with necessary data
    expected_output =  # expected output for this test
    assert np.allclose(ray_tracing(x, y, cal, mm), expected_output)

    # Test Case 3
    x, y = -2, 2
    cal = # create Variable for cal with necessary data 
    mm = # create an object for mm with necessary data
    expected_output = # expected output for this test
    assert np.allclose(ray_tracing(x, y, cal, mm), expected_output)
    
    # Add more cases if necessary