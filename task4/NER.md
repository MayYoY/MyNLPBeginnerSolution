# NER

## CRF

考虑序列的顺序信息

定义一个特征函数集合，用它来为一个标注序列打分，并据此选出最靠谱的标注序列

设一个简单的特征函数 $f_j(s, i, l_i, l_{i - 1}), g_k(s, i, l_i)$，它只考虑输入的句子 $s$，单词 $i$ 和序列对它的标注 $l_i$，以及对前一个单词的标注 $l_{i - 1}$。对集合中的这种函数进行赋权，得到对序列标注 $l$ 的评分：a matrix of transition scores
$$
sore(l|s) = \sum_{j}\sum_{i = 1}^n \lambda_jf_j(s, i, l_{i}, l_{i - 1}) + \sum_{k}\sum_{i=1}^n\mu_kg_k(s, i, l_i)
$$
对序列标注集合 $L$，由这组特征函数求得的序列标注的概率值即为：
$$
p(l|s) = \frac{\exp(score(l|s))}{\sum_{l'\in L}\exp(score(l'|s))}
$$
**每一个 HMM 模型都等价于某个 CRF**

CRF 可以定义种类更丰富的特征函数，可以使用任意的权重

## Neural Architectures for Named Entity Recognition

**两种结构**：使用 CRF 输出标记；用一种 transition-based algorithm 对输入显式构造块，并进行标记

**LSTM-CRF**：

* 首先双向 LSTM 对 word embedding 编码，再通过一个 hidden layer

* 解码不直接使用 softmax，使用 CRF

* 一个 named entity 可以跨越多个 token，一般将其划分为 IOB begginning, innside, outside 结构，但这种划分只限定了 I-label 后不能有其它 entity label；改用 **IOBES 结构**，S 表示单个 token 代表的 entity (singleton entity)，E 代表一个 entity 的结束；这样划分限定了 I-label 后只能是 I-label or E-label

**Transition-Based Chunking Model**：直接赋予多个 token 命名 constructs representations of the multi-token names

字符级表示：用 BLSTM，前向 LSTM 偏向于得到后缀的特征表示，反向 LSTM 则偏向于得到前缀；将这两级 LSTM 的最后一个输出 concatenate 就得到

## End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF

输入 char embedding 至 CNNs 得到 char representation，和 word embedding concatenate 起来作为 BLSTM 的输入，得到输出使用 CRF 解码

CRF 的势函数 $\exp(W_{y',y}^Tz + b_{y',y})$，计算概率 $p(\boldsymbol y|\boldsymbol z;\bold W, \bold b)$；训练与推理使用 **Viterbi algorithm**

Dropout

使用字符级的特征的原因在于命名实体 have orthographic or morphological evidence

## 复现

数据读入：对单词、句子的 padding

网络：

* CNNs/BLSTM 得到 character representation，concat 上 word embedding，输入 BLSTM，再由 CRF 解码
* word embedding 需要微调，取 100 维；character embedding 取 30 维

CRF 的实现：DP，Viterbi 算法

IBOES？

训练算法：梯度截断，学习率变化

evaluation：各种指标

