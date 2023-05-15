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