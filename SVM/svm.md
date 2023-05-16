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

## svm背后的最优化问题

点到直线的距离

(x,y)到Ax+By+C = 0的距离 $\frac{|Ax+By+C|}{\sqrt{A^2+B^2}}$

拓展到n维空间 $\theta^Tx_b = 0$转化为$w^Tx+b=0$

点到n维空间直线的距离为$\frac{|w^Tx+b|}{||w||}$

$$
||w|| = \sqrt{w_1^2 + w_2^2 + \cdots + w_n^2}
$$

$$
\begin{cases}
  \frac{w^Tx^{(i)} + b}{||w||} \geq d & \qquad \forall y^{(i)} = 1\\  \\
  \frac{w^Tx^{(i)} + b}{||w||} \leq -d &\qquad \forall y^{(i)} = -1\\
\end{cases} \\

\begin{cases}
  \frac{w^Tx^{(i)} + b}{||w||d} \geq 1 & \qquad \forall y^{(i)} = 1\\  \\
  \frac{w^Tx^{(i)} + b}{||w||d} \leq -1 &\qquad \forall y^{(i)} = -1\\
\end{cases}
\\
\\ d是样本点到决策边界的最短距离
\\
\begin{cases}
  w_d^Tx^{(i)} + b_d \geq 1 & \qquad \forall y^{(i)} = 1\\  \\
  w_d^Tx^{(i)} + b_d \leq -1 & \qquad \forall y^{(i)} = -1\\  
\end{cases}
\\w_d^T = \frac{w^T}{||w||d} \qquad b_d = \frac{b}{||w||d}

\\
最终推导最佳的决策边界直线是 w_d^Tx^{(i)} + b_d = 0

重命名变量 让 w^T 替代 w_d^T \qquad b 替代 b_d

\begin{cases}
  w^Tx^{(i)} + b \geq 1 & \qquad \forall y^{(i)} = 1\\  \\
  w^Tx^{(i)} + b \leq -1 & \qquad \forall y^{(i)} = -1\\  
\end{cases}

\\
转换为公式 \qquad
y^{(i)}(w^Tx^{(i)} + b) \geq 1

\\
对于任意支撑向量x，求 \max\frac{|w^Tx+b|}{||w||} = \max\frac{1}{||w||} 相当于 min||w||,等同于 min\frac{1}{2}||w||^2
$$

$$
问题转化为 求满足y^{(i)}(w^Tx^{(i)} + b) \geq 1条件下， min \frac{1}{2}||w||^2
$$
实际上是个有条件的最优化问题

## Soft Margin 和 SVM 的正则化

Soft Margin SVM 具有容错性的SVM模型

$$
min \frac{1}{2}||w||^2 \\
s.t. \qquad y^{(i)}(w^Tx^{(i)} + b) \geq 1 - \zeta _i  \qquad \zeta _i \geq 0  (对每个y_i都有一个\zeta _i)\\ 

min(\frac{1}{2}||w||^2 + \sum^m_{i=1} \zeta_i) \\

引入超参数C加入L1正则 , min(\frac{1}{2}||w||^2 + C\sum^m_{i=1} \zeta_i) \\

引入超参数C加入L2正则 , min(\frac{1}{2}||w||^2 + C\sum^m_{i=1} \zeta_i^2) \\

\zeta_i \geq 0\\
C 趋近于正无穷，即是\zeta_i逼近于0，从而退化为Hard Margin SVM,C越小意味着有更大的容错空间\zeta_i
$$

## 什么是核函数

SVM解决的关键问题是
$$
min(\frac{1}{2}||w||^2 + C\sum^m_{i=1} \zeta_i) \\

s.t. \qquad y^{(i)}(w^Tx^{(i)} + b) \geq 1 - \zeta_i \qquad \zeta_i \geq 0
$$
转换为
$$
\max\sum_{i = 1}^m\alpha_i - \frac{1}{2}\sum_{i = 1}^m\sum_{j = 1}^m\alpha_i\alpha_jy_iy_jx_ix_j \\
s.t. \qquad 0\leq\alpha_i\leq C \\
\sum_{i = 1}^m\alpha_iy_i = 0
$$

核函数（kernel function）是一种用于机器学习和模式识别中的函数。它通常用于非线性分类和回归等问题中。核函数能够在原始空间中的非线性分类问题中工作，并且不需要显式地进行特征映射到高维空间。

在支持向量机（SVM）算法中，核函数是关键的部分，它用于将数据映射到高维空间，并且使得数据可以更容易地被分类。通过核函数，我们可以在低维空间中计算出高维空间中的内积，从而避免了显式地进行特征映射的计算。

常见的核函数包括线性核函数、多项式核函数、高斯核函数等，不同的核函数选择会对分类或回归的性能产生影响。

### 多项式核函数

$$
K(x,y) = (x\cdot y + 1)^2\\
=(\sum_{i = 1}^n x_i \cdot y_i + 1)^2
= \sum_{i = 1}^n (x_i^2) (y_i^2) + \sum_{i = 2}^n\sum_{j = 1}^{i-1}(\sqrt{2}x_ix_j)(\sqrt{2}y_iy_j) + \sum_{i = 1}^n(\sqrt{2}x_i)(\sqrt{2}y_i) + 1\\
=x^`\cdot y^` \\

其中 x^` = (x_n^2,\dotsb,x_1^2,\sqrt{2}x_nx_{n-1},\dotsb,\sqrt{2}x_n,\dotsb,\sqrt{2}x_1, 1)\\

y^` = (y_n^2,\dotsb,y_1^2,\sqrt{2}y_ny_{n-1},\dotsb,\sqrt{2}y_n,\dotsb,\sqrt{2}y_1, 1)\\

对样本进行了转换,使x添加了多项式元素
$$
$$
多项式核函数K(x, y) = (x\cdot y + c)^d \\
线性核函数K(x,y) = x\cdot y
$$

### 高斯核函数

K(x,y)重新定义x和y的点乘

$$ 

K(x,y)=e^{-\gamma||x-y||^2} \\

对照高斯函数 g(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} \\

也被称为RBF核函数，Radial Basis Function Kernel，径向基函数\\

高斯核函数本质将每一个样本点映射到一个无穷维的特征空间

$$

#### 多项式特征

依靠升维使得原本线性不可分的数据线性可分

## SVM思路解决回归问题

在margin范围内包含的点越多越好，与分类问题相反（margin区域内点越少越好）