# CNN 文本分类

## 模型方法

基本方法：在 word embedding 后加一层卷积层

设嵌入词向量 $\mathbf x_i \in \mathbb R^k$，句子由 $n$ 个词构成(必要时 padding)，则句子表示为 $\mathbf{x}_{1: n}=\mathbf{x}_{1} \oplus \mathbf{x}_{2} \oplus \ldots \oplus \mathbf{x}_{n} \in \mathbb R^{n \times k}$，这就和图像类似了

使用卷积核 $\mathbf w \in \mathbb R^{h \times k}，h = 3, 4, 5$对句子矩阵卷积，得到 $\mathbf c\in \mathbb R^{n - h + 1}$，通过设置输出通道数(100)获得多张特征图；然后在*时间维度*上最大池化操作 -> 可处理变长输入

使用多个不同 size 的卷积核，再进行最大池化，得到不同的标量 -> 组成向量，输入全连接层

正则化方面，在最终的全连接层前使用 dropout	$p = 0.5$；同时采用梯度截断，最大值为 3

batch_size = 50	Adadelta