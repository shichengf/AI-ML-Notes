---
title: 贝叶斯推断与变分推断
description: 从后验为什么难算出发，理解变分推断为什么会出现在 VAE 中。
tags:
  - ml/bayesian-inference
  - ml/variational-inference
  - dl/vae
lang: zh-CN
status: growing
publish: true
created: 2026-08-28
---

# 贝叶斯推断与变分推断

这篇笔记只解决一个问题：

> 已经看到数据 $x$，但真正关心的是生成它的隐藏原因 $z$。怎样推断 $z$，又怎样训练这个生成模型？

VAE 正好包含这个问题。理解它以后，先验、后验和变分推断就不再是彼此分离的名词。

## 先分清 $z$、$\theta$ 和 $\phi$

| 符号 | 在 VAE 中表示什么 |
| --- | --- |
| $x$ | 已经观察到的数据。例如一张手写数字图片。 |
| $z$ | 单个样本背后的潜变量。例如字体粗细、倾斜程度和数字类别等隐藏因素。 |
| $\theta$ | 生成模型的参数。在 VAE 中，它们主要是 decoder 神经网络的权重。 |
| $\phi$ | 推断模型的参数。在 VAE 中，它们主要是 encoder 神经网络的权重。 |

$z$ 和 $\theta$ 的角色完全不同。每张图片都有自己的 $z$，而整个数据集共用一套 $\theta$。训练时会更新 $\theta$。给定一张新图片时，我们想推断它对应的 $z$。

### $p_\theta$ 到底是什么意思

$p_\theta$ 不是 $p$ 乘以 $\theta$。它表示“由参数 $\theta$ 控制的概率分布”。

例如，decoder 接收 $z$，输出一张图片的均值 $\mu_\theta(z)$。我们可以定义

$$
p_\theta(x\mid z)
=\mathcal{N}\left(x;\mu_\theta(z),\sigma^2 I\right).
$$

这条公式的意思是：

> 给定潜变量 $z$，decoder 先算出它应该生成什么样的图片。真实图片 $x$ 允许在这个结果附近有一些噪声。

改变神经网络权重 $\theta$，函数 $\mu_\theta(z)$ 会改变，整个条件分布 $p_\theta(x\mid z)$ 也会改变。

$p_\theta(z\mid x)$ 则表示同一个生成模型所隐含的真实后验。它回答的是另一个方向的问题：

> 已经看到图片 $x$，哪些 $z$ 最可能生成它？

这里的 $p_\theta(z\mid x)$ 不是我们已经拥有的另一个神经网络。它是由先验 $p(z)$ 和 decoder $p_\theta(x\mid z)$ 通过贝叶斯公式共同决定的分布。

## 从生成方向看，事情很简单

VAE 假设数据按照下面的过程产生：

$$
z\sim p(z),\qquad x\sim p_\theta(x\mid z).
$$

通常把 $p(z)$ 设为标准正态分布。生成一张新图片时，只需要：

1. 从 $p(z)$ 采样一个 $z$。
2. 把 $z$ 输入 decoder。
3. 根据 $p_\theta(x\mid z)$ 生成 $x$。

这个方向从隐藏原因走向观测结果，所以容易执行。

训练时遇到的是反方向。数据集只给了我们 $x$，没有告诉我们生成每个 $x$ 的 $z$。我们需要计算后验

$$
p_\theta(z\mid x)
=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}.
$$

分子给每个候选 $z$ 一个分数：

$$
p_\theta(x\mid z)p(z).
$$

$p(z)$ 检查这个 $z$ 原本是否合理。$p_\theta(x\mid z)$ 检查这个 $z$ 能否通过 decoder 解释当前的 $x$。两项都高的 $z$ 应该获得较高的后验概率。

## 后验为什么难处理

难点在分母

$$
p_\theta(x)
=\int p_\theta(x\mid z)p(z)\,dz.
$$

它需要把所有可能的 $z$ 对 $x$ 的解释都加起来，然后才能把分子归一化。

如果 $z$ 只有三个可能值，对应的未归一化分数是 $2$、$5$ 和 $3$，事情很简单。把它们除以总和 $10$，后验就是 $0.2$、$0.5$ 和 $0.3$。

VAE 中的 $z$ 通常是连续的多维向量。即使只有 20 个维度，也有无限多个可能值。decoder 又是非线性神经网络，所以这个高维积分通常没有可以直接计算的公式。

这造成两个具体问题：

1. 对一个给定的 $z$，我们可以计算 $p_\theta(x\mid z)p(z)$。但我们不知道所有候选 $z$ 的总分，因此得不到归一化后的后验密度。
2. 训练 decoder 需要提高数据的概率 $\log p_\theta(x)$。这个量本身也包含同一个难算的积分。

知道未归一化分数，并不等于能够直接得到相互独立的后验样本。MCMC 可以只使用未归一化密度，但它通常需要为每个 $x$ 运行一段采样过程。VAE 希望用神经网络对大量数据进行快速训练，因此需要另一种办法。

## 变分推断做了什么

变分推断不再强求直接计算 $p_\theta(z\mid x)$。它另外构造一个容易计算和采样的分布

$$
q_\phi(z\mid x),
$$

并让它逼近真实后验。

在 VAE 中，$q_\phi(z\mid x)$ 就是 encoder 定义的分布。常见选择是对角高斯：

$$
q_\phi(z\mid x)
=\mathcal{N}\left(
z;\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

encoder 的权重是 $\phi$。输入一张图片 $x$ 后，它输出这张图片对应的 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$。因此，每个 $x$ 都会得到自己的高斯分布，但所有图片共用同一个 encoder 和同一套参数 $\phi$。

理想目标是

$$
\min_\phi
D_{\mathrm{KL}}\left(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
\right).
$$

它的直觉很直接：

> 调整 encoder 的参数 $\phi$，让 encoder 给出的分布尽量接近生成模型真正隐含的后验。

这里使用 Reverse KL。我们可以从 $q_\phi(z\mid x)$ 采样，也可以计算它的密度。关于两个 KL 方向的差别，可以参考 [[notes/kl-divergence|KL 散度]]。

## 不知道真实后验，怎样最小化 KL

把贝叶斯公式写成对数形式：

$$
\log p_\theta(z\mid x)
=\log p_\theta(x,z)-\log p_\theta(x).
$$

代入 KL：

$$
\begin{aligned}
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p_\theta(z\mid x)\right)
&=\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log q_\phi(z\mid x)-\log p_\theta(x,z)
\right]
+\log p_\theta(x).
\end{aligned}
$$

移项以后得到

$$
\log p_\theta(x)
=\mathcal{L}_{\mathrm{ELBO}}(x)
+D_{\mathrm{KL}}\left(
q_\phi(z\mid x)\parallel p_\theta(z\mid x)
\right),
$$

其中

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

这组公式表达了三个重要事实：

1. $\log p_\theta(x)$ 是我们真正想提高的数据概率。
2. KL 是近似后验与真实后验之间的误差，而且不会小于零。
3. ELBO 等于数据概率减去这个误差，所以它是 $\log p_\theta(x)$ 的下界。

真实后验出现在关系式中，但计算 ELBO 只需要 $q_\phi(z\mid x)$、$p(z)$ 和 $p_\theta(x\mid z)$。这三个量我们都能计算或采样。因此，我们可以最大化 ELBO，绕开难算的 $p_\theta(x)$ 和 $p_\theta(z\mid x)$。

## 为什么 VAE 必须用到变分推断

VAE 想学习一个带潜变量的生成模型

$$
p_\theta(x,z)=p_\theta(x\mid z)p(z).
$$

如果能够精确计算后验，我们可以推断每个样本的 $z$，也可以直接优化数据似然。但神经网络 decoder 让后验和边缘似然通常无法精确计算。

变分推断同时解决了这两个问题：

| VAE 需要什么 | 变分推断提供什么 |
| --- | --- |
| 根据 $x$ 推断潜变量 $z$ | encoder 给出近似后验 $q_\phi(z\mid x)$ |
| 训练生成模型 $p_\theta(x\mid z)$ | ELBO 提供一个可以计算的训练目标 |

变分推断中的 “variational” 指的是在一组候选分布中寻找最合适的分布。在 VAE 中，这个候选分布由 encoder 和参数 $\phi$ 表示。“autoencoder” 则来自 encoder 与 decoder 的网络结构。

因此，VAE 不是在普通 autoencoder 上随意添加一个 KL 项。它先是一个后验难算的潜变量生成模型，然后才用 encoder 和 ELBO 把推断与训练变成可计算的问题。

### VAE 的目标为什么有两项

利用

$$
p_\theta(x,z)=p_\theta(x\mid z)p(z),
$$

ELBO 可以整理成

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=
\underbrace{
\mathbb{E}_{q_\phi(z\mid x)}
\left[\log p_\theta(x\mid z)\right]
}_{\text{让 }z\text{ 能解释 }x}
-
\underbrace{
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p(z)\right)
}_{\text{让编码分布接近先验}}.
$$

第一项通常被叫作 reconstruction term。更准确地说，它是 decoder 对观测 $x$ 给出的期望对数似然。只有选择特定的观测分布以后，它才会变成熟悉的 MSE 或 binary cross entropy。

第二项让每个样本的编码分布不要随意散落在潜空间中。训练完成后，我们希望从统一的先验 $p(z)$ 采样并生成合理数据。如果 encoder 把样本编码到先验几乎不会出现的区域，从先验采样时就很难落到这些区域。

### 一次训练具体发生什么

对于一张图片 $x$：

1. Encoder 计算 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$。
2. 采样 $\epsilon\sim\mathcal N(0,I)$，再令

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
$$

3. Decoder 用 $z$ 计算 $p_\theta(x\mid z)$。
4. 使用 ELBO 同时更新 $\phi$ 和 $\theta$。

第二步是 reparameterization trick。随机性被移到 $\epsilon$，所以梯度仍然可以通过 $z$ 传回 encoder。

这里的 encoder 不是在记住一张图片对应的固定 $z$。它学习的是一个从任意 $x$ 到近似后验分布的共享映射。这叫 amortized inference。训练完成后，新样本只需要经过一次 encoder，就能得到近似后验，不必重新运行一轮迭代推断。

## 最后把角色连起来

| 对象 | 作用 |
| --- | --- |
| $p(z)$ | 规定生成时从哪里采样潜变量。 |
| $p_\theta(x\mid z)$ | Decoder。把潜变量变成数据分布。 |
| $p_\theta(z\mid x)$ | 生成模型隐含的真实后验。我们想要它，但通常算不出来。 |
| $q_\phi(z\mid x)$ | Encoder。用容易处理的分布近似真实后验。 |
| ELBO | 不直接计算真实后验，也能同时训练 encoder 和 decoder 的目标。 |

变分推断的核心不是“用一个高斯替代另一个分布”这么简单。它把一个难以计算的后验推断问题，变成了一个可以用梯度下降训练的优化问题。VAE 则用 encoder 把这套推断过程做成了一次神经网络前向计算。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)
