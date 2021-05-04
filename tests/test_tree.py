import xgboost as xgb
from sklearn.datasets import load_boston
import json


if __name__ == '__main__':
    boston = load_boston()
    data = boston['data']
    X = data[:, :-1]
    y = data[:, -1]
    dm = xgb.DMatrix(X, label=y)

    params = {
        'tree_method': 'exact', 'verbosity': 0,
        'objective': 'reg:squarederror',
        'eta': 1,
        'lambda': 0,
        'alpha': 0.1,
        'max_depth': 3
    }
    bst = xgb.train(params, dm, 10)
    trees = bst.get_dump()
    model_path = 'test_model_json_io.json'
    bst.save_model(fname=model_path)
    with open(model_path, 'r') as fd:
        j_model = json.load(fd)
    print(0)


