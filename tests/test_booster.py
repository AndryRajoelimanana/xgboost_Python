from gbm.booster import train
from sklearn import datasets
import xgboost as xgb
import numpy as np


def test_gbtree(x, y):
    params = {'num_parallel_tree': 5, 'updater': 'grow_colmaker',
              'colsample_bylevel': 0.5}
    bst = train(params, x, labels=y, num_boost_round=5)

    # tree0_dict = bst.get_tree(0)

    # xgboost
    dm = xgb.DMatrix(x, label=y)
    params = {'num_parallel_tree': 5, 'tree_method': 'exact',
              'colsample_bylevel': 0.5}
    bst1 = xgb.train(params, dm, num_boost_round=5)

    pred_xgboost = bst1.predict(dm, output_margin=True, training=True)
    pred_own = bst.predict(x, output_margin=True, training=True)

    assert np.all(np.isclose(pred_xgboost, pred_own[:, 0])), 'some values ' \
                                                             'differs'

    print(f'Test done GBTree')


def test_gblinear(x, y):
    params = {'booster': 'gblinear', 'learning_rate': 1.0, 'eta': 1.0}
    bst = train(params, x, labels=y, num_boost_round=10)

    # xgboost
    dm = xgb.DMatrix(x, label=y)
    bst1 = xgb.train(params, dm, num_boost_round=10)
    pred_xgboost = bst1.predict(dm, output_margin=True, training=True)
    pred_own = bst.predict(x, output_margin=True, training=True)

    assert np.all(np.isclose(pred_xgboost, pred_own[:, 0])), 'some values ' \
                                                             'differs'

    print(f'Test done GBLinear')


if __name__ == '__main__':
    boston = datasets.load_boston()
    data0 = boston['data']
    X0 = np.array(data0[:, :-1])
    y0 = np.array(data0[:, -1])

    test_gbtree(X0, y0)
    test_gblinear(X0, y0)
