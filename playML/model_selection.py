import numpy as np

def train_test_split(X, y, test_ratio = 0.2, seed = None):
  """将数据X和y按照test_ratio分割成X_train，X_test, y_train, y_test"""
  
  assert X.shape[0] == y.shape[0], \
    "the size of X must be equal to the size of y"
  assert 0.0 <= test_ratio <= 1.0, \
    "test_ratio must be vallid"
    
  if seed:
    np.random.seed(seed)
  
  shuffled_indexes = np.random.permutation(len(X))
  
  test_size = int(test_ratio * len(X))
  test_indexs = shuffled_indexes[:test_size]
  train_indexs = shuffled_indexes[test_size:]
  
  X_train = X[train_indexs]
  y_train = y[train_indexs]
  
  X_test = X[test_indexs]
  y_test = y[test_indexs]
  return X_train, X_test, y_train, y_test