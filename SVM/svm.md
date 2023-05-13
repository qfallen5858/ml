# 什么是SVM

支撑向量机 support vector machine

既可以解决分类问题，也可以解决回归问题

不适定问题：决策边界不唯一

```
SVM尝试寻找一个最有的决策边界，距离两个类别的最近的样本最远，

最近的样本就是支撑向量
```

SVM要最大化margin， margin就是 距离*2

svm解决的是线性可分问题， hard margin SVM

Soft Margin SVM, 解决线性不可分问题时，适当改进