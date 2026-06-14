import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x, dtype=int)
    p = np.array(p, dtype=float)
    
    if x.shape != p.shape:
        raise ValueError("Shapes error")

    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("NO Prob match")

    return float(np.sum(x*p))
