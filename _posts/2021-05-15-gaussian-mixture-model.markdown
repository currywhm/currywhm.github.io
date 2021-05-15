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
<center class="half">
其中<img src="/img/in-post/gaussian_model/qiwang.png"/>为数据均值(期望)，<img src="/img/in-post/gaussian_model/sd.png"/>为数据标准差(Standard deviation).</center>  
当样本数据X是多维数据十，高斯分布遵从一下概率密度函数:
<img src="/img/in-post/gaussian_model/gaussian_2.png"/>
其中，<img src="/img/in-post/gaussian_model/qiwang.png"/>为数据均值(期望)，<img src="/img/in-post/gaussian_model/cov.png"/>为协方差，D为数据维度。  
**高斯混合模型**  




