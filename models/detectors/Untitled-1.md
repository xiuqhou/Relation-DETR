二元交叉熵有两种形式。

- 第一种形式：

    $$
    \mathcal L_\text{bce}=\left\{
    \begin{aligned}
    &-\log(q)&,p=1\\
    &-\log(1-q)&,p=0
    \end{aligned}
    \right.
    $$

    由于$p$非0即1，我们可以将他们合并成一个式子：
    $$
    \mathcal L_\text{bce}=-p\log(q)-(1-p)\log(1-q)
    $$

    其中$p$和$q$分别是GT和预测值。

- 另外一种形式：

    $$
    \mathcal L_\text{bce}=-\log(p_t), p_t=\left\{
    \begin{aligned}
    &q&,p=1\\
    &1-q&,p=0\\
    \end{aligned}
    \right.
    $$

    由于$p$非0即1，我们同样可以将$p_t$的表达式合并成一个式子：

    $$
    p_t=pq+(1-p)(1-q)
    $$

    这就是我们论文中采用的形式。

这两种表达式其实就是FocalLoss论文中的公式(1)和公式(2).

> Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2980-2988.


为什么他俩都是二元交叉熵？我们可以验证一下$p$在0和1情况下两种表达式的结果：

当$p=0$的时候:

- 第一种形式中，当$p=0$的时候第一项$-p\log(q)=0$，第二项系数$(1-p)=1$，因此二元交叉熵变成了$-\log(1-q)$
- 第二种形式中$p_t=1-q$，因此二元交叉熵变成了$-\log(1-q)$

可以看出$p=0$的时候这两个表达式时相等的，同样也可以验证$p=1$的时候他俩是相等的。

因此结论是，当作为分类的损失函数时用哪种形式来表示都可以，我们在论文中是沿用了FocalLoss和FocusDETR中第二种形式定义的损失。但上面验证的基本假设是$p$非0即1，当以salience_supervision作真值时，$p$变成了[0, 1]之间的连续值，而不是非0即1的离散值，这时候其实他们两个是不等价了，因此论文的表示确实是有问题，是我们的纰漏。不过代码实现中我们是直接调用F.binary_cross_entropy_with_logits来计算二元交叉熵，所以对比结果和结论是没问题的。

> 代码可以参考：https://github.com/xiuqhou/Salience-DETR 中的 models/bricks/losses.py 文件中的 sigmoid_focal_loss 方法。