### 核方法(kernel method)



> 在阅读迁移学习论文的时候，感觉自己kernel method方面不是很清楚，所以观看了李振轩老师的《kernel method》视频课程[B站链接](https://www.bilibili.com/video/BV1hW411C7ny?p=1)。用python实现了课程中提到的各种算法。

[TOC]



#### 1 基本想法

利用某个函数$\phi$,将不可线性分割的feature向量映射到可以线性分割的高维空间中。

例如下面的例子，本来需要椭圆曲线分开的二维平面的点，经过$\phi$映射到三维中，就可以用一个平面分割了。
$$
\phi : R^2 \rightarrow R^3 \\ (x_1,x_2) \rightarrow (z_1,z_2,z_3) = (x_1^2,\sqrt{2}x_1x_2,x_2^2)
$$

<center class = "half">
<img src = “./basic_2d.png”  width = “50%” align = left><img src = “./basic_3d.png”  width = “50%” align = right>
</center>





对于这一个$\phi$,可以证明映射后的内积等于映射前的内积的平方。$$<\phi(x),\phi(x')> = (<x,x'>)^2 = K(x,x') \\ K : kernel function$$

