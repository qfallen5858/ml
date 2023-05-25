from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
count = 0
for name in Path(r'e:/ai/train_upload/images/').glob('*.bmp'):
  img = plt.imread(name)
  plt.imshow(img)
  with open(r'E:/ai/train_upload/labelTxt_new/' + name.stem + '.txt') as f:
    lines = f.readlines()
    for line in lines:
      numbers = line.split(',')
      x1= float(numbers[0])
      y1 = float(numbers[1]), 
      x2=float(numbers[2]),
      y2=float(numbers[3]), 
      x3=float(numbers[4]), 
      y3=float(numbers[5]),
      x4=float(numbers[6]),
      y4=float(numbers[7])
      plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], alpha=0.3, color='r')
  plt.title(name.stem)
  plt.show()
  count += 1
  if count > 10:
    break  
  