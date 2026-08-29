---
title: KL 散度
description: KL 散度的方向、优化含义和一些有用的性质。
tags:
  - ml/information-theory
  - dl/vae
lang: zh-CN
status: seedling
publish: true
created: 2026-08-28
---

# KL 散度

## 定义与符号

设 $P$ 是目标分布，$Q$ 是用来近似它的另一个分布。对于离散随机变量，KL 散度定义为

$$
D_{\mathrm{KL}}(P \parallel Q)
= \sum_x p(x) \log \frac{p(x)}{q(x)}
= \mathbb{E}_{x \sim P}\left[\log p(x)-\log q(x)\right].
$$

### 怎么读这条公式

这里补充两个不太直观的记号：

| 符号 | 它表示什么 |
| --- | --- |
| $\mathbb{E}_{x\sim P}$ | 对所有结果求加权平均。下标 $x\sim P$ 表示 $x$ 按照 $P$ 分布，也表示平均时由 $P$ 提供权重。 |
| $Q_\theta$ 或 $q_\phi$ | $\theta$ 和 $\phi$ 表示控制分布的模型参数。训练模型就是调整这些参数。 |

对于连续随机变量，求和变成积分：

$$
D_{\mathrm{KL}}(P \parallel Q)
= \int p(x) \log \frac{p(x)}{q(x)}\,dx.
$$

这里的期望由 $P$ 加权。因此，$P$ 中概率较高的区域会获得更高权重。如果 $p(x)>0$ 而 $q(x)=0$，KL 散度会变成无穷大。

## Forward KL 与 Reverse KL

设 $P$ 是目标分布，$Q$ 是近似分布。两个方向分别是

$$
\underbrace{D_{\mathrm{KL}}(P\parallel Q)}_{\text{Forward KL}}
=\mathbb{E}_{x\sim P}\left[\log\frac{p(x)}{q(x)}\right],
$$

以及

$$
\underbrace{D_{\mathrm{KL}}(Q\parallel P)}_{\text{Reverse KL}}
=\mathbb{E}_{x\sim Q}\left[\log\frac{q(x)}{p(x)}\right].
$$

把比值反过来会改变对数内部的符号。如果期望仍然由 $P$ 加权，结果只会是 Forward KL 的负数，并不会成为另一个散度。Reverse KL 还把加权分布从 $P$ 换成了 $Q$。这两个变化共同造成了不同的行为。

> [!important] 核心直觉
>
> | 方向 | 主要惩罚 | 典型行为 |
> | --- | --- | --- |
> | Forward KL，$D_{\mathrm{KL}}(P\parallel Q)$ | **漏掉 $P$ 中真实存在的区域** | 覆盖更多模式 |
> | Reverse KL，$D_{\mathrm{KL}}(Q\parallel P)$ | **把概率分配给 $P$ 认为不真实的区域** | 寻找单个模式 |

对于 Forward KL，如果某个区域的 $p(x)$ 很大，而 $q(x)$ 接近零，代价会很大。漏掉 $P$ 的一个模式甚至可能让散度变成无穷大。如果某个区域的 $p(x)$ 很小，即使 $q(x)$ 相对较大，它获得的权重仍然很低。因此，更广的覆盖通常很便宜，但不是完全免费。$Q$ 的总概率仍然必须等于 1，多出来的概率需要从其他区域拿走。

对于 Reverse KL，如果 $Q$ 把概率放在 $p(x)$ 接近零的区域，代价会很大。但是，如果 $q(x)$ 在 $P$ 的某个模式上接近零，这个被遗漏的模式在 $Q$ 的期望中几乎没有权重。当 $Q$ 无法表示所有模式时，它可能只选择一个高密度区域。

因此，Forward KL 常被称为 mass-covering，Reverse KL 常被称为 mode-seeking。这些名称描述的是受限分布 $Q$ 近似固定目标 $P$ 时的常见行为，不是所有优化问题中的普遍保证。

### 通常应该优化哪个方向

当我们可以从目标分布 $P$ 采样时，Forward KL 很自然。它可以写成

$$
D_{\mathrm{KL}}(P\parallel Q)
=\mathbb{E}_{x\sim P}[\log p(x)]
-\mathbb{E}_{x\sim P}[\log q(x)].
$$

第一项不依赖 $Q$。来自 $P$ 的样本足以估计第二项。这就是最大似然和监督学习中的交叉熵通常采用 Forward KL 的原因。它们会强烈惩罚数据分布概率很高，但模型概率很低的情况。

Reverse KL 适合另一种计算条件。假设我们可以从 $Q$ 采样，而目标分布只能通过未归一化密度表示。

“未归一化”表示我们可以为每个候选值计算一个相对分数 $\widetilde p(x)$，但还没有把所有分数转换成总和为 1 的概率。

> [!example] 一个简单例子
> 假设三个候选值的分数分别是 2、5 和 3。把它们除以总和 $Z=10$，才能得到真正的概率 0.2、0.5 和 0.3。即使暂时不知道 $Z$，我们仍然知道第二个候选值比第一个更合理，因为 $5>2$。

如果候选值有几百万个，或者 $x$ 是连续的高维变量，计算所有分数的总和或积分可能非常困难。这时我们可以计算某个给定 $x$ 的 $\widetilde p(x)$，却无法轻松得到归一化常数 $Z$。

贝叶斯后验就是一个常见例子。对于给定的潜变量 $z$，我们可以计算

$$
\widetilde p(z)=p(x\mid z)p(z),
$$

但要得到真正的后验 $p(z\mid x)$，还需要对所有可能的 $z$ 求和或积分。这个难以计算的总量就是归一化常数。

一般地，归一化后的目标分布可以写成：

$$
p(x)=\frac{\widetilde{p}(x)}{Z}.
$$

此时

$$
D_{\mathrm{KL}}(Q\parallel P)
=\mathbb{E}_{x\sim Q}[\log q(x)-\log\widetilde{p}(x)]
+\log Z.
$$

未知的归一化常数 $\log Z$ 不依赖 $Q$。因此，我们可以通过从 $Q$ 采样来优化 Reverse KL，不需要先解决从 $P$ 采样这个困难问题。

> [!tip] 实用判断
> 如果有来自目标分布 $P$ 的样本，可以考虑 Forward KL。如果容易从近似分布 $Q$ 采样，而且目标 $P$ 可以计算到一个归一化常数，那么 Reverse KL 通常更方便。

| 场景 | 常见方向 | 这个方向为什么方便 |
| --- | --- | --- |
| 最大似然与监督学习 | Forward，$D_{\mathrm{KL}}(P_{\mathrm{data}}\parallel Q_\theta)$ | 数据集提供来自 $P_{\mathrm{data}}$ 的样本。 |
| 知识蒸馏 | Forward，$D_{\mathrm{KL}}(P_{\mathrm{teacher}}\parallel Q_{\mathrm{student}})$ | 教师模型提供目标概率。 |
| 变分推断 | Reverse，$D_{\mathrm{KL}}(Q_\phi\parallel P_{\mathrm{posterior}})$ | 可以从 $Q_\phi$ 采样，也可以计算未归一化的后验。 |
| 变分自编码器 | Reverse，$D_{\mathrm{KL}}(q_\phi(z\mid x)\parallel p_\theta(z\mid x))$ | ELBO 避免了直接从难处理的真实后验采样。 |
| KL 正则化强化学习与 RLHF | Reverse 形式，$D_{\mathrm{KL}}(\pi_\theta\parallel\pi_{\mathrm{ref}})$ | 样本来自学习中的策略，参考模型对相同输出进行评分。 |

Reverse KL 的经典例子可以参考 [[notes/bayesian-and-variational-inference|贝叶斯推断与变分推断]]。PPO 本身不应该被看作 Reverse KL 的另一个名称。

### 有用的性质

#### 为什么 KL 不会是负数

函数 $-\log t$ 是凸函数。假设在 $p(x)>0$ 的地方都有 $q(x)>0$，使用 Jensen 不等式可以得到

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=\mathbb{E}_{x\sim P}\left[-\log\frac{q(x)}{p(x)}\right] \\
&\geq -\log\mathbb{E}_{x\sim P}\left[\frac{q(x)}{p(x)}\right] \\
&= -\log\sum_{x:p(x)>0}q(x) \\
&\geq 0.
\end{aligned}
$$

最后一步成立，是因为 $Q$ 在 $P$ 的支持集上的概率总和不会超过 1。等号成立需要 $q(x)/p(x)$ 是常数，归一化以后只能有 $P=Q$。如果存在 $p(x)>0$ 但 $q(x)=0$ 的位置，KL 会变成正无穷，仍然不会是负数。

#### 为什么代码还是会给出负数

| 负数来自哪里 | 它意味着什么 |
| --- | --- |
| 单个样本或 token | 当 $q(x)>p(x)$ 时，$\log p(x)-\log q(x)$ 可以是负数。只有它在 $P$ 下的完整期望才是 KL。 |
| 有限样本的 Monte Carlo 平均 | 估计量存在方差，所以可能暂时小于零。增加样本通常会减小这种波动。 |
| 未归一化的目标分布 | $\mathbb{E}_Q[\log q-\log\widetilde p]$ 等于 Reverse KL 减去 $\log Z$。没有补回这个常数时，结果可以是负数。 |
| ELBO 实现 | 训练代码经常把 $-D_{\mathrm{KL}}$ 加入需要最大化的目标。日志里记录的可能是 KL 的负数。 |
| 浮点数误差 | 本应接近零的值可能略小于零，低精度计算或两个接近的 log probability 相减时更常见。 |
| 密度实现错误 | 缺少归一化、漏掉变量变换的 Jacobian，或者 reduction 不一致，都会让计算结果不再是真正的 KL。 |

所以看到负数时，先检查代码记录的是否真的是完整、精确、已经归一化的 KL。较小的 Monte Carlo 负估计可能合理，但闭式 KL 公式通常不应小于零，除非存在很小的浮点数误差。

KL 还满足数据处理不等式。对两个分布应用相同的确定性或随机变换，不会增加它们之间的 KL。较粗糙的观察可以隐藏分布之间的差异，但不会制造新的差异。

对于联合分布，KL 可以进行链式分解：

$$
D_{\mathrm{KL}}(P(X,Y)\parallel Q(X,Y))
=D_{\mathrm{KL}}(P(X)\parallel Q(X))
+\mathbb{E}_{x\sim P(X)}\left[
D_{\mathrm{KL}}(P(Y\mid x)\parallel Q(Y\mid x))
\right].
$$

这个性质可以把序列级别的 KL 分解成条件分布上的项，因此适合分析自回归模型和策略。

当 $P$ 和 $Q$ 已经很接近时，Forward KL 与 Reverse KL 在局部也很接近。如果 $q(x)=p(x)+\varepsilon(x)$，$\int\varepsilon(x)\,dx=0$，并且 $\varepsilon$ 相对 $p$ 很小，那么两个方向具有相同的二阶近似：

$$
D_{\mathrm{KL}}(P\parallel Q)
\approx D_{\mathrm{KL}}(Q\parallel P)
\approx \frac{1}{2}\int\frac{\varepsilon(x)^2}{p(x)}\,dx.
$$

当两个分布相距较远、support 不同，或者包含多个模式时，它们的区别才会明显。

## 与交叉熵和似然的关系

$P$ 与 $Q$ 之间的交叉熵是

$$
H(P,Q)=-\sum_x p(x)\log q(x),
$$

而 $P$ 的熵是

$$
H(P)=-\sum_x p(x)\log p(x).
$$

它们满足

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P \parallel Q).
$$

当 $P$ 是固定的目标分布时，$H(P)$ 是常数。改变 $Q$ 只会改变 KL 项。因此，对 $Q$ 最小化交叉熵等价于最小化 $D_{\mathrm{KL}}(P\parallel Q)$。如果模型能够表示 $P$，最小值在 $Q=P$ 时取得。如果模型能力受限，训练会选择当前模型族中 KL 最小的 $Q$。

对于观测样本 $x_1,\ldots,x_N$，模型的似然是

$$
L(\theta)=\prod_{i=1}^{N}q_\theta(x_i).
$$

平均负对数似然是交叉熵的样本估计：

$$
-\frac{1}{N}\log L(\theta)
=-\frac{1}{N}\sum_{i=1}^{N}\log q_\theta(x_i).
$$

当数据分布固定时，最大化似然、最小化负对数似然、最小化交叉熵和最小化 Forward KL 是同一个优化问题。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Lecture Notes on Statistics and Information Theory](https://web.stanford.edu/class/stats311/lecture-notes.pdf)
- [Challenges and Opportunities in High-dimensional Variational Inference](https://arxiv.org/abs/2103.01085)
