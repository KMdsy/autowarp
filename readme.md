# Read me file

论文“Autowarp: Learning a Warping Distance from Unlabeled Time Series Using Sequence Autoencoders” (NIPS18)
的tensorflow实现。**此代码仅为个人练习所用，非官方版本。**

由于该文章没有开源代码，因此做了一个实现。

本代码是根据原文的Supplementary Materials中Section B的伪代码编写的。代码中的变量尽量使用论文公式中的变量，以增加可读性。
尽量所有的代码都有原文对应，原文内容均标注于代码之前。

> 论文详情
>
> Abid, A., & Zou, J. Y. (2018). Learning a warping distance from unlabeled time series using sequence autoencoders. 
> In Advances in Neural Information Processing Systems (pp. 10547-10555).
> 
> @inproceedings{abid2018learning,
>  title={Learning a warping distance from unlabeled time series using sequence autoencoders},
>  author={Abid, Abubakar and Zou, James Y},
>  booktitle={Advances in Neural Information Processing Systems},
>  pages={10547--10555},
>  year={2018}
> }

## 存在的问题

1. 根据dynamic programing的原理，我发现原文Supplementary Materials中Lemma 1下的距离计算公式可能有误，应将公式中的n, m分别替换为i, j。
我试图向原作者寻求确认，但并未得到回复。

2. 在soft-dtw中，Jacobian Matrix是使用dynamic programming计算的，且有明确的公式说明相关的计算。然而在Autowarp中作者并未说明梯度的计算过程。
由于本人实在才疏学浅且时间紧迫，不得不放弃公式推导，而使用tensorflow进行自动梯度计算。

3. 由于整个dynamic programming均由tensorflow完成，涉及诸多for循环，导致计算图的规模庞大无比，**因此本代码存在以下两个不可避免的问题：**

    1. 非常吃内存，因为涉及非常多的循环，每个循环中都手动添加了大量的计算操作。

    2. 运行非常缓慢。


**如果您在使用此代码的时候，对上述存在的问题有更好的解决方法，真诚的希望您提交issues，或者联系我：shaoyu1122@foxmail.com**


## 使用方法

1. run train.py: 导入一个训练集，并学习该训练集数据的隐空间表达，然后训练autowarp中的三个参数——alpha，gamma，epsilon

2. run clustering.py：将1中训练得到的三个参数在该文件头进行设置，然后在相应的数据集上进行聚类实验。performance metric: NMI, clustering algorithm: K-means。

## Requirements

tensorflow >= 2.2.0
