"""
Trevor Amestoy
July 2023
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_test

def rodionov_regimes(data, l, p):
    """Implements Rodionov's (2004) regime shift detection algorithm:

    Rodionov, S. N. (2004). A sequential algorithm for testing climate regime shifts. 
    Geophysical Research Letters, 31(9).

    Args:
        data (array): Timeseries array of std values
        l (int): The assumed minimum regime length
        p (float): The singificance probability to use when assessing shifts

    Returns:
        list, list: Two lists: The regime-shift indices, the RSI values 
    """
    # Step 1: Set the cut-off length l of the regimes
    # l: Number of years for each regime
    # p: Probability level for significance
    n = len(data)
    regime_shift_indices = []
    rsi = np.zeros(n)

    # Step 2: Determine the difference diff for statistically significant mean values
    t_stat = np.abs(t_test.ppf(p, (2*l-2)))
    avg_var = np.mean([np.var(data[i:(i+l)]) for i in range(n-l)])
    diff = t_stat * np.sqrt(2 * avg_var / l)

    # Step 3: Calculate initial mean for regime R1
    r1 = np.mean(data[:l])
    r1_lower = r1 - diff
    r1_upper = r1 + diff

    i = l + 1
    while i < n:
        
        # Step 4: Check if the value exceeds the range of R1 ± diff
        if data[i] < r1_lower or data[i] > r1_upper:
            j = i

            # Estimate the mean of regime 2 as the upper bound of r1 distribution
            test_r2 = r1_lower if data[i] < r1_lower else r1_upper
            
            # Step 5: Calculate regime shift index (RSI) across next l-window
            for k in range(j + 1, min(j + l,n)):
                if data[j] > r1:
                    rsi[j] += (data[k] - test_r2)/(avg_var*l)
                elif data[j] < r1:
                    rsi[j] += (test_r2 - data[k])/(avg_var*l)

                # Step 6: Test for a regime shift at year j
                if rsi[j] < 0:
                    rsi[j] = 0
                    break

            # Step 7: Confirm significant regime shift at year j
            if rsi[j] > 0:
                regime_shift_indices.append(j)
                r2 = np.mean(data[j:min(j+l,n)])
                r1 = r2
                r1_lower = r1 - diff
                r1_upper = r1 + diff

            i = j + 1
        else:
            # Recalculate average for R1 to include the new value
            r1 = ((l - 1) * r1 + data[i]) / l

        i += 1
    return regime_shift_indices, rsi