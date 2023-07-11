# Work Notes 2023.5.23

​	MMD（最大均值差异）是迁移学习，尤其是Domain adaptation （域适应）中使用最广泛（目前）的一种损失函数，主要用来度量两个不同但相关的分布的距离。两个分布的距离定义为：
![image-20230523155935321](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230523155935321.png)

其中 H表示这个距离是由 ϕ ( )  将数据映射到再生希尔伯特空间（RKHS）中进行度量的。
	Domain adaptation的目的是将源域（Source domain）中学到的知识可以应用到不同但相关的目标域（Target domain）。本质上是要找到一个变换函数，使得变换后的源域数据和目标域数据的距离是最小的。所以这其中就要涉及如何度量两个域中数据分布差异的问题，因此也就用到了MMD。

​	MMD的关键在于如何找到一个合适的 ϕ ( ) 来作为一个映射函数。但是这个映射函数可能在不同的任务中都不是固定的，并且这个映射可能高维空间中的映射，所以是很难去选取或者定义的。那如果不能知道ϕ，如何求MMD：

![image-20230523163835649](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230523163835649.png)

所以MMD又可以表示为：
![image-20230523163926036](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230523163926036.png)

在大多数论文中（比如DDC, DAN），都是用高斯核函数来作为核函数，至于为什么选用高斯核，最主要的应该是高斯核可以映射无穷维空间
![image-20230523170250673](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230523170250673.png)

**计算MMD**

```python
import pandas as pd
import numpy as np

# 读数据
reviews1 = pd.read_csv('analysis results/desert_snow_cosin(-1,1)_120.csv')
reviews2 = pd.read_csv('analysis results/desert_itself_cosin(-1,1).csv')
reviews3 = pd.read_csv('analysis results/z_desert_snow_cosin(-1,1).csv')
reviews4 = pd.read_csv('analysis results/z_desert_itself.csv')

# 得到数据分布
hist1, _ = np.histogram(reviews1['desert_snow'], bins=50, density=True)
hist2, _ = np.histogram(reviews2['desert_itself'], bins=50, density=True)
hist3, _ = np.histogram(reviews3['z_desert_snow'], bins=50, density=True)
hist4, _ = np.histogram(reviews4['z_desert_itself'], bins=50, density=True)


# 计算MMD
def gaussian_kernel(x, y, sigma=1.0):
    gamma = 1.0 / (2 * sigma ** 2)
    pairwise_dists = np.sum(np.square(x), axis=1).reshape(-1, 1) + np.sum(np.square(y), axis=1) - 2 * np.dot(x, y.T)
    return np.exp(-gamma * pairwise_dists)


def mmd(hist_1, hist_2, sigma=1.0):
    K11 = gaussian_kernel(hist_1.reshape((-1, 1)), hist_1.reshape((-1, 1)), sigma)
    K22 = gaussian_kernel(hist_2.reshape((-1, 1)), hist_2.reshape((-1, 1)), sigma)
    K12 = gaussian_kernel(hist_1.reshape((-1, 1)), hist_2.reshape((-1, 1)), sigma)
    MMD = np.sqrt(K11.mean() + K22.mean() - 2 * K12.mean())
    return MMD


mmd1_3 = mmd(hist1, hist3, sigma=1.0)
mmd2_4 = mmd(hist2, hist4, sigma=1.0)
print('mmd1_3:', mmd1_3)
print('mmd2_4:', mmd2_4)
```

```
mmd1_3: 0.4087326121547762
mmd2_4: 0.18386505255672472
```

![image-20230605093659866](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230605093659866.png)
