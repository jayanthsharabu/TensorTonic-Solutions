import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)
    _, cnts = np.unique(y, return_counts=True)
    n = len(y)
    if n == 0:
        return 0.0
    ps = cnts * (1/n)
    ps = ps[ps > 0]
    
    return float(np.sum(-ps * np.log2(ps)))
    
    