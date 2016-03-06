---
title: ADMM调研
date: 2016-02-23 17:30:31
tags:
---

## ADMM调研

### 1. 约束优化问题一般解决方案

#### 1.1. **Dual Ascent（对偶上升法）**

对于凸函数的优化问题，对偶上升法对偶上升法核心思想就是引入一个对偶变量，然后利用交替优化的思路，使得两者同时达到optimal。

一个典型的等式约束最优化问题，形式化表示如下：

$$
\begin{align}
&\min_{x} \quad f(x) \\
& s.t. \; Ax=b
\end{align}
$$

目标函数是\\(f(x)\\)，\\(x=(x_1,x_2,\cdots,x_n)^T \in R^n\\) （\\(n\\)表示参数向量个数）；下面是等式约束。

引入拉格朗日乘子（又称算子），这里用\\(\beta\\)表示乘子，得到的拉格朗日公式为：

$$\mathcal{L}(x,\beta) = f(x) + \beta^T (Ax-b) \qquad(1.2)$$

对偶函数：
	
$$
	g(\beta) = \inf_{x} L(x,\beta) \qquad(1.3)
	$$
	
在强对偶性假设下，即最小化原凸函数（primal）等价于最大化对偶函数（dual），两者会同时达到optimal。可得：
	
$$
	x^{\*} = \arg \min_{x} \mathcal{L}(x, y^{\*})  \qquad(1.4)
	$$
	
如果对偶函数\\(g(\beta)\\)可导，使用Dual Ascent法，交替更新参数，使得同时收敛到最优。迭代公式如下：
	
$$
\begin{align}
x^{k+1} & := \arg \min_{x} L(x,\beta^{k}) \quad（x-最小化）\\
\beta^{k+1} & := \beta^{k} + \alpha^k \nabla g(\beta) = y^k + \alpha^k(A x^{k+1} -b) \quad (对偶变量更新，\alpha^k为步长)
\end{align} \qquad(1.5)
$$
	
Dual Ascent法要求目标函数\\(f(x)\\)为**强凸函数**（一般的目标函数难以满足）。

> 强凸函数
	> 
	> 函数\\(f: I \rightarrow R\\) 成为强凸的，若\\(\exists \alpha > 0\\)，使\\(\forall (x,y) \in I \times I, \forall t \in [0,1]\\)，恒有：
	>
	
$$
f[t x + (1-t)y] \le tf(x) + (1-t) f(y) - t(1-t)\alpha(x-y)^2
$$

#### 1.2. Dual Decomposition


Dual Ascent的缺陷就是它对目标函数的限制过于严格。但是它有一个非常好的性质：

**当目标函数\\(f\\)可分（separable）时，整个问题可以拆解成多个子问题，分块优化后得到局部参数，然后汇集起来整体更新全局参数。非常有利于并行化处理。**

形式化表示：

$$
\begin{align}
& \min_{x} \quad f(x) = \sum_{i=1}^{m} f_i(x_i) \\
& s.t. \; Ax=\sum_{i=1}^{m} A_ix_i = b
\end{align}		\qquad(1.6)
$$

拉格朗日函数：
 
$$
\mathcal{L}(x,\beta) = \sum_{i=1}^{m} \mathcal{L}_i(x_i, \beta) = \sum_{i=1}^{m} \left(f_i(x_i) + \beta^T A_i x_i - \frac{1}{N} \beta^T b \right) \qquad(1.7)
$$

对应的迭代公式：

$$
\begin{align}
x_{i}^{k+1} & := \arg \min_{x} L_i(x_i,\beta^{k}) \quad（多个x_i并行最小化步）\quad(1)\\
\beta^{k+1} & := \beta^{k} + \alpha^k \nabla g(\beta) = y^k + \alpha^k(A x^{k+1} -b) \quad (汇集整体的x,对偶变量更新) \quad(2)
\end{align} \qquad\qquad(1.8)
$$

#### 1.3. 扩展拉格朗日乘子法

dual ascent方法对于目标函数要求比较苛刻，为了放松假设条件，同时比较好优化，于是就有了Augmented Lagrangians方法，目的就是放松对于\\(f(x)\\)严格凸的假设和其他一些条件，同时还能使得算法更加稳健。

具体做法：原有的拉格朗日公式添加惩罚函数项：
	
$$
\mathcal{L}_{\rho}(x,\beta) = f(x) + \beta^T (Ax-b) + \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2 \qquad(1.9)
$$
	
公式解读：
	
>$$
\mathcal{L}_{\rho}(x,\beta) = f(x) + \overbrace{ \underbrace{\beta^T (Ax-b)}_{拉格朗日乘子法} + \underbrace{ \frac{\rho}{2} {\Vert Ax-b \Vert}_2^2}_{函数惩罚法} }^{增强拉格朗日乘子法} \qquad(n.1.1)
$$	

参数迭代公式

$$
\begin{align}
x^{k+1} & := \arg \min_{x} L(x,\beta^{k}) \quad（x-最小化）\\
\beta^{k+1} & := \beta^{k} + \alpha^k \nabla g(\beta) = y^k + \rho(A x^{k+1} -b) \quad (对偶变量更新，\alpha^k为步长)
\end{align} \qquad(1.10)
$$

虽然Augmented Lagrangians方法有优势，但也破坏了dual ascent方法的利用分解参数来并行的优势。当\\(f\\)是separable时，对于Augmented Lagrangians却是not separable的（因为平方项写成矩阵形式无法用之前那种分块形式）



### 2. Alternating Direction Method of Multipliers (ADMM)

#### 2.1. ADMM概述

为了整合dual ascent可分解性与method multiplers优秀的收敛性质，人们就又提出了改进形式的优化ADMM。目的就是想能分解原函数和扩增函数，以便于在对\\(f(x)\\)更一般的假设条件下并行优化。

ADMM从名字可以看到是在原来Method of Multipliers加了个Alternating Direction，可以大概猜想到应该是又想引入新变量，然后交叉换方向来交替优化。形式如下：
		
$$
\begin{align}
& min \quad f(x) + g(z)  \\
& s.b \quad Ax + B z = C
\end{align}  \qquad (2.1)
$$
	
> 其中\\(x \in R^n, z \in R^m; A \in R^{p \times n}, B \in R^{p \times m}, C \in R^p\\)。
	
增强Lagrange函数

$$
\mathcal{L}_{\rho}(x,z,\beta) = f(x) + g(z) + \underline{ \beta^T(Ax+Bz-C) + \frac{\rho}{2} {\Vert Ax+Bz-C \Vert}_2^2 }  \qquad(2.2)
$$
	
从上面形式确实可以看出，ADMM的思想就是想把primal变量、目标函数拆分，但是不再像dual ascent方法那样，将拆分开的\\(x_i\\)都看做是xx的一部分，后面融合的时候还需要融合在一起，而是最先开始就将拆开的变量分别看做是不同的变量xx和zz，同时约束条件也如此处理，这样的好处就是后面不需要一起融合xx和zz，保证了前面优化过程的可分解性。于是ADMM的优化就变成了如下序贯型迭代（这正是被称作alternating direction的缘故）：

#### 2.2. 参数迭代公式（缩放形式）

定义残差：\\(\underline{r = Ax+Bz-C}\\)，令 \\(\underline{\mu = \frac{1}{\rho}\beta} \in R^p\\)（对偶变量归一化）. 增强Lagrange函数等价于：
	
$$
\mathcal{L}_{\rho}(x,z,r,\mu) = f(x) + g(z) + \underline{ \frac{\rho}{2} {\Vert r + \mu \Vert}_2^2 - \frac{\rho}{2} {\Vert \mu \Vert}_2^2 }  \qquad(2.3)
$$
	
> 推导如下：
>
$$
	\begin{align}
	\beta^T(Ax+Bz-C) + \frac{\rho}{2} {\Vert Ax+Bz-C \Vert}_2^2 & = \beta^T \cdot r + \frac{\rho}{2} {\Vert r \Vert}_2^2 \\
	& = \frac{\rho}{2} {\left \Vert  r + \frac{1}{\rho} \beta \right \Vert}_2^2 - \frac{1}{2\rho} {\Vert \beta \Vert}_2^2 \\
	& = \frac{\rho}{2} {\Vert r + \mu \Vert}_2^2 - \frac{\rho}{2} {\Vert \mu \Vert}_2^2
	\end{align}
	$$	
	
ADMM迭代公式转化为：
	
$$
\begin{cases}
	x^{k+1} = \arg \min_{x} \left( f(x) + \frac{\rho}{2} {\Vert Ax^k + Bz^k - C + \mu^k \Vert}_2^2 \right) \qquad (1) \\
	z^{k+1} = \arg \min_{z} \left( g(z) + \frac{\rho}{2} {\Vert Ax^{k+1} + Bz^k - C + \mu^k \Vert}_2^2 \right) \qquad(2)\\
	\mu^{k+1} = \mu^k + Ax^{k+1} + Bz^{k+1} - C \qquad\qquad\qquad(3)
	\end{cases} \qquad(2.4)
	$$

+ 公式(2.4)的理解

	典型的利用ADMM分布式求解的问题中，

	+ 公式（1）用于各部分数据的**局部参数更新**；
	+ 公式（2）用于将个部分得到的局部优化参数综合成**全局的参数**；
	+ 公式（3）用于**对偶变量的更新**，是使得整个迭代过程稳定和高效率的关键。

#### 2.3. 参数迭代公式推导

+ **\\(f(x)\\)为二次函数式时**

> 例如：\\(f(x) = \frac{1}{2} x^T P x + q^T x + r\\)。损失函数为平方损失时，符合这一场景。


令\\(\underline{-v = Bz^k - C + \mu^k}\\)，对参数\\(x\\)求偏导：

$$
\begin{align}
\frac{\partial{ \left( f(x) + \frac{\rho}{2} {\Vert Ax - v \Vert}_2^2 \right) }} {\partial{x}} 
& = \frac{\partial{ \left( \frac{1}{2} x^T P x + q^T x + r + \frac{\rho}{2} {\Vert Ax - v \Vert}_2^2 \right) }} {\partial{x}} \\
& = \frac{\partial{(\frac{1}{2} x^T P x)}}{\partial{x}} + \frac{\partial{(q^T x + r)}}{\partial{x}} + \underline{ \frac{\rho}{2} \cdot \frac{(Ax)^T(Ax) - 2(Ax)^Tv + {\Vert v \Vert}_2^2} {\partial{x}} } \\
& = Px + (q^T)^T + \underline { \frac{\rho}{2} \left( 2A^TAx - 2A^Tv\right)} \\
& = Px + q + \underline{ \rho A^TAx - \rho A^T v } \\
& = (P + \rho A^TA)x + (q - \rho A^T v) = 0
\end{align}  \qquad(2.5)
$$

偏导数为0，得到参数\\(x\\)的迭代公式：

$$
x = (P + \rho A^TA)^{-1} \cdot (\rho A^T v - q) \quad (v中含有参数z和\mu) \qquad(2.6)
$$

> 

+ **\\(f(x)\\)为norm 1范数形式时**
	
	> 例如：\\(f(x) = \lambda {\Vert x \Vert}_{1} = \lambda (|x_1| + |x_2| + \cdots + |x_n|)\\)
	
	对参数\\(x\\)求偏导：
	
	+ (1). 当\\(\frac{\partial {f(x)}} {\partial{x}} = \lambda\\)时

	$$
	\begin{align}
	\frac{\partial{ \left( f(x) + \frac{\rho}{2} {\Vert Ax - v \Vert}_2^2 \right) }} {\partial{x}} 
	& = \frac{\partial{\left( \lambda{\Vert x \Vert}_1 + \frac{\rho}{2} {\Vert Ax - v \Vert}_2^2\right)}} {\partial{x}} \\
	& = \frac{\partial{(\lambda {\Vert x \Vert}_1)}} {\partial{x}} + \underline{ \frac{\rho}{2} \cdot \frac{(Ax)^T(Ax) - 2(Ax)^Tv + {\Vert v \Vert}_2^2} {\partial{x}} } \\
	& = \lambda I + \underline{ \rho A^TAx - \rho A^T v } \\
	& = \rho A^TAx + (\lambda I - \rho A^T v) = 0
	\end{align} \qquad(2.7)
	$$
	
	取\\(A=I\\)，则有
	
	$$
	x^{*} = I_{n \times p} v - \frac{\lambda}{\rho} I_{n \times 1} > 0  \qquad(2.7-1)
	$$
	
	+ (2). 当\\(\frac{\partial {f(x)}} {\partial{x}} = -\lambda\\)时

	$$
	\begin{align}
	\frac{\partial{ \left( f(x) + \frac{\rho}{2} {\Vert Ax - v \Vert}_2^2 \right) }} {\partial{x}} 
	& = \frac{\partial{\left( \lambda{\Vert x \Vert}_1 + \frac{\rho}{2} {\Vert Ax - v \Vert}_2^2\right)}} {\partial{x}} \\
	& = \frac{\partial{(\lambda {\Vert x \Vert}_1)}} {\partial{x}} + \underline{ \frac{\rho}{2} \cdot \frac{(Ax)^T(Ax) - 2(Ax)^Tv + {\Vert v \Vert}_2^2} {\partial{x}} } \\
	& = -\lambda I + \underline{ \rho A^TAx - \rho A^T v } \\
	& = \rho A^TAx - (\lambda I + \rho A^T v) = 0
	\end{align} \qquad(2.8)
	$$
	
	取\\(A=I\\)，则有
	
	$$
	x^{*} = I_{n \times p} v + \frac{\lambda}{\rho} I_{n \times 1} < 0 \qquad(2.8-1)
	$$
	
	+ 软阈值（Soft Thresholding）

		综合(1)和(2)可得，参数\\(x^{*}\\)形式为：
		
		$$
		x^{*} = S_{\frac{\lambda}{\rho}}(v) \rightarrow S_{a}(v) = (v-a)_{+} - (-v-a)_{+} = 
		\begin{cases}
		v-a, & \quad if \quad v \ge a \\
		\quad 0, &\quad if \quad -a < v < a \\
		v+a, & \quad if \quad v \le -a
		\end{cases} \qquad(2.9)
		$$
		
		> \\(a = \frac{\lambda}{\rho}\\) 是常数
		> 
		> [软阈值概念与示例](http://blog.csdn.net/abcjennifer/article/details/8572994)

### 3. ADMM for Lasso Model

#### 3.1. Lasso问题与ADMM

基于\\(l1\\)正则的线性回归（即Lasso模型）的目标函数，形式化表示：
	
$$
\min_{w} \; \frac{1}{2} {\Vert Xw - Y\Vert}_2^2 + \lambda {\Vert w \Vert}_1 \qquad(3.1)
	$$
	
> 符号解释：<br>
	> \\(w \in R^n\\)：参数向量，n为向量元素个数，即特征数；<br>
	> \\(X \in R^{m \times n}\\)：表示训练数据特征矩阵,m为训练集个数；<br>
	> \\(Y \in R^m\\)：样本label组成的m维向量；<br>
	> \\(\lambda\\)：正则化因子（初始值可通过交叉验证获得）
	
+ Lasso目标函数格式："loss + regularization"

	目标函数适用ADMM框架求解形式，改写为

$$
	\begin{align}
	& \min \quad \overbrace{\frac{1}{2} {\Vert Xw - Y\Vert}_2^2}^{f(w)} + \overbrace{ \lambda {\Vert \theta \Vert}_1}^{g(\theta)} \\
	& s.b \qquad w - \theta = 0
	\end{align} \qquad(3.2)
	$$

+ 目标函数可分（separable）

	训练数据集切分为\\(L\\)个部分，分布式训练：

$$
	\begin{align}
	\frac{1}{2} \sum_{l=1}^{L} {\Vert X_{l} w_l - Y_l\Vert}^2 + \frac{\lambda}{L} \sum_{l=1}^{L} {\Vert w \Vert}_1 = \sum_{l=1}^{L} \left( \underline{ \frac{1}{2} {\Vert X_{l} w_l - Y_l\Vert}^2 + \frac{\lambda}{L} {\Vert w \Vert}_1} \right)
	\end{align} \quad(3.3)
	$$
	
这里，\\(l=1,2,\cdots,L\\)表示数据集划分为\\(L\\)个子集，\\(w_l\\)对应于第\\(l\\)部分数据上得到的参数。
	
将其改写成分布式ADMM框架可求解的形式。令\\(f_l(w_l) = \frac{1}{2} {\Vert X_{l} w_l - Y_l\Vert}^2, g_l(w) = \frac{\lambda}{L} {\Vert w \Vert}_1 \\)
	
>$$
	\begin{align}
	& \min \quad f_l(w_l) + g_l(\theta) \\
	& s.b \qquad w_l - \theta = 0
	\end{align} \quad l=1,2,\cdots,L	\qquad(n.3.2)
	$$

每部分数据，对应的增强Lagrange函数为：
	
$$
\mathcal{L}_{\rho}(w_l, \theta, \beta_l) = \frac{1}{2} {\Vert X_l w_l - Y_l\Vert}_2^2 + \frac{\lambda}{L} {\Vert \theta \Vert}_1 + \beta_l^T (w_l-\theta) + \frac{\rho}{2} {\Vert w_l - \theta \Vert}_2^2  \qquad(3.4)
$$
	
+ 参数\\(w\\)的迭代公式

	由于\\(f(w)\\)为二次函数式，即:
	
	> 
	$$
	f(w) = \frac{1}{2} {\Vert Xw - Y\Vert}_2^2 = \frac{1}{2} w^T X^T X w - Y^T Xw + \frac{1}{2} {\Vert Y \Vert}_2^2
	$$

	相应参数
	
	\\(P=X^T X, q=-X^T Y, A=I_{n \times n}, B=-I_{n \times n}, v=\theta^t - \frac{1}{\rho} \beta^t \\)。对参数\\(w\\)求偏导，得到参数\\(w\\)的迭代公式：

	$$
	\begin{align}
	w^{k+1} & = (P + \rho A^TA)^{-1} \cdot (\rho A^T v - q) \\
	& = (X^T X + \rho I)^{-1} \cdot (\rho v + X^T Y) \\
	& = (X^T X + \rho I)^{-1} \cdot \left(X^T Y + \rho(\theta^k - \frac{1}{\rho} \beta^k) \right) \\
	& = (X^T X + \rho I)^{-1} \cdot (X^T Y + \rho \theta^k - \beta^k)
	\end{align}  \qquad(3.5)
	$$
	
	
+ 参数\\(\theta\\)的迭代公式

	由于\\(g(\theta)\\)是\\(l1\\)范数形式，即：
	
	>
	$$
	g(\theta) = \lambda {\Vert \theta \Vert}_1 = \lambda (|\theta_1| + |\theta_2| + \cdots + |\theta_n|)
$$

$$
\theta^{k+1} = S_{\frac{\lambda}{\rho}}(w^{k+1}+\frac{1}{\rho}\beta^k) = 
\begin{cases}
    v-a, & \quad if \quad v \ge a \\
    \quad 0, &\quad if \quad -a < v < a \\
    v+a, & \quad if \quad v \le -a
\end{cases} \qquad(3.6)
$$

+ 参数\\(\beta\\)的迭代公式

	$$
	\beta^{k+1} = \beta^{k} + \rho(w^{k+1} - \theta^{k+1})  \qquad(3.7)
	$$


#### 3.2. ADMM分布式更新参数过程

这里给出参数迭代公式：

$$
\begin{align}
w_l^{k+1} &= \arg \min_{w} \left( f_l(w) + \frac{\rho}{2} {\Vert w + \theta^k + \mu_l^k \Vert}_2^2 \right) \qquad (1) \\
z^{k+1} &= \arg \min_{\theta} \left( g_l(\theta) + \frac{\rho}{2} {\Vert \theta - \overline{w^{k+1}} - \overline{\mu^k} \Vert}_2^2 \right) \quad(2)\\
\mu_l^{k+1} &= \mu_l^k + w_l^{k+1} - \theta^{k+1} \qquad\quad\qquad\qquad\qquad(3)
\end{align} \qquad(3.8)
$$

分布式环境下执行过程：

+ 首先，再每个数据分块上，分别执行(1)中对应的更新，得到该数据块上更新后的参数（迭代过程）。这一步是分布式进行的，而且各个数据块之间不需要通信；
+ 然后，根据各部分更新得到的局部参数，执行公式（2）得到综合以后的整体参数\\(\theta\\)；
+ 最后，根据公式(3)更新对偶变量\\(\mu\\)（\\(\beta的归一化\\)），并将更新后的整体参数\\(\theta\\)和\\(\mu\\)分发至各个数据块的处理单元。

#### 3.3. 分布式解决方案

+ MapReduce
	+ Mapper：（1） 
	+ Reducer：（2），（3）
	+ 代码示例：[admm for hadoop](https://github.com/intentmedia/admm.git)
+ MPI： 提供Allreduce和Broadcast操作，用于机器之间的通信
	+ 计算单元：（1） 
	+ Allreduce：(2)
	+ Broadcast: (3)
+ Rabit：
	+ 仅包含MPI的一个Allreduce子集，提供容错；
	+ 运行在Yarn上，避免MPI和Hadoop之间的数据传输；
	+ 计算过程同MPI；
	+ 与DMLC强耦合
	+ 示例：[opticlick-admm@baigang](http://10.210.228.76/opticlick/admm/tree/master)
+ Spark
	+ 单结点：(1)
	+ treeAggregate: (2), 相当于Allreduce
	+ （全局）广播变量 
	+ 代码示例：[admm for spark](https://github.com/dieterichlawson/admm)

#### 3.4. ADMM适用范围

+ 目标函数结构为"loss + regularation"
+ 目标函数可分：分布式求解
+ 在ML中以loss function是平方损失的模型，都可以用ADMM求解。
