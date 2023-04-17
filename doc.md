# 算法

## 超参数和模型参数

- 超参数： 在算法运行前需要决定的参数
- 模型参数： 算法过程中学习的参数
  
```

kNN算法没有模型参数
kNN算法中的k是典型的超参数
解决分类问题
```

- 明可夫斯基距离
- 曼哈顿距离
- 欧拉距离

## 线性回归算法

- 解决回归问题
- 思想简单，实现容易
- 许多强大的非线性模型的基础
- 结果具有很好的可解释性
- 蕴含机器学习中的很多重要思想

简单线性回归：样本特征只有一个

损失函数：模型中没有拟合的那部分 

效用函数：模型中拟合的那部分

通过分析问题，确定问题的损失函数或者效用函数；通过最优化损失函数或者效用函数，获得机器学习的模型，近乎所有参数学习算法都是这样的套路，如：

- 线性回归
- 多项式回归
- 逻辑回归
- SVM
- 神经网络
- 等等

最小二乘法：最小化误差的平方

### 回归算法评估指标

#### 均方误差MSE（Mean Squared Error）

$$
\frac{1}{m} \sum_{i=1}^m (y_{test}^{(i)} - \hat{y}_{test}^{(i)})^2
$$

#### 均方根误差RMSE(Root Mean Squared Error), MSE的平方根

$$
\sqrt{\frac{1}{m} \sum_{i=1}^m (y_{test}^{(i)} - \hat{y}_{test}^{(i)})^2} 
$$

#### 平均绝对误差MAE（Mean Absolute Error）

$$
\frac{1}{m} \sum_{i=1}^m \ |y_{test}^{(i)} - \hat{y}_{test}^{(i)}|
$$

#### R Squared

最好的指标是R Squared
$$
R^2 = 1-\frac{\sum_i (\hat{y}^{(i)}-y^{(i)})^2}{\sum_i (\bar{y}^{(i)}-y^{(i)})^2}  = 1-\frac{(\sum_i^m (\hat{y}^{(i)}-y^{(i)})^2)/m}{(\sum_i^m (\bar{y}^{(i)}-y^{(i)})^2)/m} = 1- \frac{MSE(\hat{y}, y)}{Var(y)}
\\
\hat{y}：预测值 \qquad
\bar{y}:均值
\\
\sum_i (\hat{y}^{(i)}-y^{(i)})^2 是使用模型预测产生的误差  
\\
\sum_i (\bar{y}^{(i)}-y^{(i)})^2 是使用 y=\bar{y} 预测产生的误差，baseline Model  
$$

- R^2 <=1
- R^2 越大越好。当我们的预测模型不犯任何错误时，R^2取得最大值1
- 当我们的模型等于基准模型时,R^2为0
- 如果R^2<0，说明学习到的模型还不如基准模型。此时，很有可能是数据不存在任何线性关系

### 多元线性回归

$$
求解y= \theta_0 + \theta_1x1+\theta_2x2 + \dots + \theta_nx_n
$$
公式变化
$$
\hat{y}^{(i)}= \theta_0X_0^{(i)} + \theta_1X1^{(i)}+\theta_2X2^{(i)} + \dots + \theta_nX_n^{(i)}， \theta_0X_0^{(i)} = 1
$$

$$
\hat{y}^{(i)}= X^{(i)}\cdot\theta
$$
转换成矩阵
$$ 
X_b = \left(
  \begin{matrix}
    1 & X_1^{(1)} & X_2^{(1)} & \dots & X_n^{(1)} \\
    1 & X_1^{(2)} & X_2^{(2)} & \dots & X_n^{(2)} \\
    \cdots & & & & \cdots \\
    1 & X_1^{(m)} & X_2^{(m)} & \dots & X_n^{(m)} \\
  \end{matrix}
\right) \qquad 
\theta= \left(
  \begin{matrix}
    \theta_0 \\
    \theta_1 \\
    \cdots \\
    \theta_n
  \end{matrix}
\right)
\\
\hat{y} = X_b \cdot \theta
\\
目标：使 \sum_{i=1}^m(y^{(i)} - \hat{y}^{(i))^2}尽可能小
\\
转换为使(y-X_b\cdot\theta)^T(y-X_b\cdot\theta)尽可能小
\\ \qquad \\
推导出多元线性回归的正规方程解\\
\theta = (X^T_bX_b)^{-1}X_b^Ty
$$
问题： 时间复杂度高：O(n^3)(优化为O(n^2.4))

优点：不需要对数据做归一化处理

