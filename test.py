import numpy as np
# import matplotlib.pyplot as plt
# from math import sqrt
# from collections import Counter
# from sklearn.neighbors import KNeighborsClassifier
from knn import KNNClassifier


kNN_classifier = KNNClassifier(k=6)



raw_data_X = [[3.3935, 2.3312],
              [3.11007 , 1.7815],
              [1.3438, 3.36836],
              [3.58229,4.678179],
              [2.28036,2.86699],
              [7.42343,4.6965],
              [5.74505,3.53398],
              [9.17216,2.5111],
              [7.79278,3.4241],
              [7.9398,0.791637]
              ]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

kNN_classifier.fit(X_train, y_train)

x = np.array([8.0936, 3.3657])
X_predict = x.reshape(1, -1)
predict_y = kNN_classifier.predict(X_predict)
# predict_y = kNN_classify(6, X_train, y_train, x)
print(predict_y)
# # distance = []
# # for x_train in X_train:
# #   d = sqrt(np.sum((x-x_train) ** 2))
# #   distance.append(d)

# distances = [sqrt(np.sum((x-x_train) ** 2)) for x_train in X_train]

# nearest = np.argsort(distances)

# k = 6

# topK_y = [y_train[i] for i in nearest[:k]]

# print(topK_y)

# votes = Counter(topK_y)

# predict_y = votes.most_common(1)[0][0]

# plt.scatter(X_train[y_train==0,0], X_train[y_train==0, 1], color='green', marker="o")
# plt.scatter(X_train[y_train==1,0], X_train[y_train==1, 1], color='red', marker="+")
# if predict_y == 1:
#   plt.scatter(x[0], x[1], color='red', marker="x")
# else:
#   plt.scatter(x[0], x[1], color='green', marker="x")


# plt.show()