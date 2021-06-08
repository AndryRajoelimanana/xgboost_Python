# xgboost ML written in Python


This is a pure python version of all algorithms (steps) in the xgboost 
package (originally written in C++). All randomizations used are the same as 
in the original package (Mersenne Twister pseudo-random number generator, 
MT19937). Drawn random values and shuffling are similar to those from C++ 
standard 
`uniform_int_distribution`, `uniform_real_distribution` and `shuffle`. 



**DO NOT USE THIS** program, this was only meant to show all gradient 
boosting steps implemented in xgboost in an easy-to-read programing language 
(Python).


# Run and compare with xgboost 

## Decision Tree Boosting GBTree
``` Python
from gbm.booster import train
from sklearn import datasets
import xgboost as xgb
import numpy as np

# Loading data
boston = datasets.load_boston()
data0 = boston['data']
X0 = np.array(data0[:, :-1])
y0 = np.array(data0[:, -1])

# Train current model using boosted tree

params = {'num_parallel_tree': 5, 'updater': 'grow_colmaker',
          'colsample_bylevel': 0.5}
bst = train(params, x, labels=y, num_boost_round=5)
tree0_dict = bst.get_tree(0)


# Train original xgboost
dm = xgb.DMatrix(x, label=y)
params = {'num_parallel_tree': 5, 'tree_method': 'exact',
          'colsample_bylevel': 0.5}
bst1 = xgb.train(params, dm, num_boost_round=5)

# predict 
pred_own = bst.predict(x, output_margin=True, training=True)
pred_xgboost = bst1.predict(dm, output_margin=True, training=True)

# Compare result
assert np.all(np.isclose(pred_xgboost, pred_own[:, 0])), 'some values differ'
```

## Linear Regression Boosting (GBLinear)
Multiple rounds of gblinear is equivalent to a Lasso Regression


``` Python

# Train current model using boosted linear model
params = {'booster': 'gblinear', 'learning_rate': 1.0}
bst = train(params, x, labels=y, num_boost_round=10)

# Train original xgboost
dm = xgb.DMatrix(x, label=y)
bst1 = xgb.train(params, dm, num_boost_round=10)

# predict 
pred_own = bst.predict(x, output_margin=True, training=True)
pred_xgboost = bst1.predict(dm, output_margin=True, training=True)


assert np.all(np.isclose(pred_xgboost, pred_own[:, 0])), 'some values differ'
```