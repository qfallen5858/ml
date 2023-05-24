from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

for name in Path(r'').glob('*.bmp'):
  img = plt.imread(name)
  plt.imshow(img)
  with open(r'' + name.stem + '.txt') as f:
    lines = f.readlines()
    for line in lines:
      [x1, y1, x2, y2, x3, y3, x4, y4] = line.split(',')
      plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], alpha=0.3, color='r')
  plt.title(name.stem)
  plt.show()
  