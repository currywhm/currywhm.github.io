---
layout:       post
title:        "高斯混合模型(GMM)"
date:         2021-05-15
author:       "sipengwei"
header-mask:  0.3
catalog:      false
multilingual: false
tags:
    - gaussian model
---
**混合模型**  
混合模型是一个可以用来表示在总体分布（distribution）中含有K个子分布的概率模型，换句话说，混合模型表示了观测数据在总体中的概率分布，它是一个由K个子分布组成的混合分布。混合模型不要求观察数据提供关于子分布的信息，来计算观察数据在总体分布中的概率。

**高斯模型**  
**单高斯模型**  
当样本数据X是一维数据时，高斯分布遵从下方概率密度函数：
<img src="/img/in-post/gaussian_model/gaussian_1.png"/>  

当样本数据X是多维数据十，高斯分布遵从一下概率密度函数:
![qiwang](/img/in-post/gaussian_model/gaussian_2.png)  

**高斯混合模型**  
高斯混合模型可以看作是由 K 个单高斯模型组合而成的模型，这 K 个子模型是混合模型的隐变量（Hidden variable）。一般来说，一个混合模型可以使用任何概率分布，这里使用高斯混合模型是因为高斯分布具备很好的数学性质以及良好的计算性能。

举个不是特别稳妥的例子，比如我们现在有一组狗的样本数据，不同种类的狗，体型、颜色、长相各不相同，但都属于狗这个种类，此时单高斯模型可能不能很好的来描述这个分布，因为样本数据分布并不是一个单一的椭圆，所以用混合高斯分布可以更好的描述这个问题，如下图所示：
![qiwang](/img/in-post/gaussian_model/Hidder_var.png)
![qiwang](/img/in-post/gaussian_model/hidder_2.png)

**高斯模型参数**  
对于单高斯模型，我们可以通过最大似然法([Maximum likelihood](https://www.cnblogs.com/wjy-lulu/p/7010258.html))估算参数的值。
![pdf](/img/in-post/gaussian_model/pdf.png)
由于每个点发生的概率都很小，乘积会变得极其小，不利于计算和观察，因此通常我们用 Maximum Log-Likelihood 来计算（因为 Log 函数具备单调性，不会改变极值的位置，同时在 0-1 之间输入值很小的变化可以引起输出值相对较大的变动）：
![log](/img/in-post/gaussian_model/log.png)
对于高斯混合模型，Log-Likelihood 函数是：
![log](/img/in-post/gaussian_model/log-like.png)
如何计算高斯混合模型的参数呢？这里我们无法像单高斯模型那样使用最大似然法来求导求得使 likelihood 最大的参数，因为对于每个观测数据点来说，事先并不知道它是属于哪个子分布的（hidden variable），因此 log 里面还有求和，对于每个子模型都有未知的参数 ，直接求导无法计算。需要通过迭代的方法求解。  
**EM 算法**  
EM 算法是一种迭代算法，1977 年由 Dempster 等人总结提出，用于含有隐变量（Hidden variable）的概率模型参数的最大似然估计。  
每次迭代包含两个步骤：  
![log](/img/in-post/gaussian_model/s-step.png)
这里不具体介绍一般性的 EM 算法（通过 Jensen 不等式得出似然函数的下界 Lower bound，通过极大化下界做到极大化似然函数），只介绍怎么在高斯混合模型里应用从来推算出模型参数。  
![log](/img/in-post/gaussian_model/e-step.png)
至此，我们就找到了高斯混合模型的参数。需要注意的是，EM 算法具备收敛性，但并不保证找到全局最大值，有可能找到局部最大值。解决方法是初始化几次不同的参数进行迭代，取结果最好的那次。  

**代码实现**  
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit_predict(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
```   

经过几次高斯模型迭代以后，曲线区域拟合，拟合得分0.9以上。
![log](/img/in-post/gaussian_model/gaussian_em.png)




