import os
import pandas as pd
import requests

PATH = './iris/'

r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

with open(PATH, 'iris.data', 'w') as f:
  f.write(r.text)
