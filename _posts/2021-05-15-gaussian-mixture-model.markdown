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
[qiwang](/img/in-post/gaussian_model/hidder_2.png)sipengwei456456

**高斯模型参数**  
对于单高斯模型，我们可以通过最大似然法([Maximum likelihood](https://www.cnblogs.com/wjy-lulu/p/7010258.html))估算参数的值。  




