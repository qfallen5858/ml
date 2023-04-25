# 主成分分析 Principal Component Analysis

通常用于数理统计

- 一个非监督的机器学习算法
- 主要用于数据的降维
- 通过降维，可以发现更便于人类理解的特征
- 其他应用：可视化、去噪 

## 问题域

- 如何找到这个让样本间间距最大的轴？
- 如何定义样本间间距？

使用方差（Variance）描述样本疏密
$$
Var(x)=\frac{1}{m}\sum_{i=1}^m(x_i-\bar{x})^2
$$

找到一个轴，使得样本空间的所有点映射到这个轴后，方差最大

第一步：将样例的均值归位0(demean),所有样本减去均值
  $$
  Var(x)=\frac{1}{m}\sum_{i=1}^mx_i^2 \\
  \bar{x} = 0\\
  对所有样本进行demean处理，求一个轴的方向w=(w_1,w_2)，使得我们所有的样本，映射到w以后，有：\\
  Var(X_project)=\frac{1}{m}\sum_{i=1}^m(X_{project}^{i}-\bar{X}_{project})^2\\
  Var(X_project)=\frac{1}{m}\sum_{i=1}^m||X_{project}^{(i)}-\bar{X}_{project}||^2=\frac{1}{m}\sum_{i=1}^m||X_{project}^{(i)}||^2 \tag{向量差的模运算}
  $$
  $$
  求X^{(i)} = (X_1^{(i)},X_2^{(i)})，在向量w=(w_1,w_2)上的映射，就是计算(将)\\
  w只需用来表示方向，可以用模为1的方向向量表示\\
  X^{(i)}\cdot w = ||X^{(i)}||\cdot||w||\cos(\theta)=||X^{(i)}||\cos(\theta)=||X_{project}^{(i)}|| \\
  于是变为求Var(X_project)=\frac{1}{m}\sum_{i=1}^m||X^{(i)}\cdot w||^2\\
  $$

$$
目标：求w,使得 Var(X_project)=\frac{1}{m}\sum_{i=1}^m(X^{(i)}\cdot w)^2 最大\\

扩展成n维向量有Var(X_project)=\frac{1}{m}\sum_{i=1}^m(X_1^{(i)}\cdot w_1 + X_2^{(i)}\cdot w_2+ \dotsb + X_n^{(i)}\cdot w_n)^2 最大\\
Var(X_project)=\frac{1}{m}\sum_{i=1}^m(\sum_{j=1}^nX_j^{(i)}\cdot w_j )^2\\
一个目标函数的最优化问题，使用梯度上升法解决
$$

再调整下公式
$$
目标：求w，使得f(X)=\frac{1}{m}\sum_{i=1}^m(X_1^{(i)}\cdot w_1 + X_2^{(i)}\cdot w_2 + \dotsb + X_n^{(i)}\cdot w_n)
$$

$$
\nabla f = \Bigg( \begin{matrix}
  \frac{\partial f}{w_1}  \\
  \frac{\partial f}{w_2}  \\
  \dotsc \\
  \frac{\partial f}{w_n}  \\
\end{matrix} \Bigg)
=
\frac{2}{m}\Bigg( \begin{matrix}
  \sum_{i=1}^m(X^{(i)}\cdot w)X_1^{(i)}\\
  \sum_{i=1}^m(X^{(i)}\cdot w)X_2^{(i)}\\
  \dotsb \\
  \sum_{i=1}^m(X^{(i)}\cdot w)X_n^{(i)}
\end{matrix}
  \Bigg)
=
\frac{2}{m}\cdot X^T\cdot(Xw)
$$

求出第一主成分之后，如何求出下一个主成分？

数据进行改变，将数据在第一个主成分上的分量去掉