import xgboost as xgb
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import ctypes
from scipy.sparse import csc_matrix


c_bst_ulong = ctypes.c_uint64

boston = load_boston()
x = boston.data
y = boston.target

dmat = xgb.DMatrix(x, label=y)

x_csc = csc_matrix(x)

