"""
Based on CRF (Chinese Restaurant Franchise)  
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import numba
from scipy.special import gamma, digamma, gammaln
