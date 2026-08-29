---
title: 贝叶斯推断与变分推断
description: 先验、似然、后验与变分近似的基础关系。
tags:
  - ml/bayesian-inference
  - ml/variational-inference
  - dl/vae
lang: zh-CN
status: seedling
publish: true
created: 2026-08-28
---

# 贝叶斯推断与变分推断

## 基本问题

贝叶斯推断是一种在观察数据以后更新不确定性的框架。设 $x$ 是已经观察到的数据，$z$ 是未知量。在不同模型中，$z$ 可以是参数、潜在表示或隐藏状态。

> [!important] 核心思想
> 先验描述观察 $x$ 以前对 $z$ 的认识。后验描述利用 $x$ 中的信息以后对 $z$ 的认识。

## 先验、似然、证据与后验

| 名称 | 含义 |
| --- | --- |
| 先验，$p(z)$ | 观察 $x$ 以前，模型对 $z$ 的假设。 |
| 似然，$p_\theta(x\mid z)$ | 给定 $z$ 时，当前观测数据出现的可能性。 |
| 证据，$p_\theta(x)$ | 考虑所有可能的 $z$ 以后，模型分配给 $x$ 的总概率或概率密度。 |
| 后验，$p_\theta(z\mid x)$ | 观察 $x$ 以后，对 $z$ 得到的更新分布。 |

贝叶斯公式把它们连接起来：

$$
p_\theta(z\mid x)
=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}.
$$

证据是

$$
p_\theta(x)
=\int p_\theta(x\mid z)p(z)\,dz.
$$

分子同时考虑两件事。一个 $z$ 在观察数据以前是否合理，以及它能否解释当前的 $x$。分母负责让后验积分为 1。

## 为什么后验很难处理

困难通常来自证据 $p_\theta(x)$。计算它需要对所有可能的 $z$ 进行积分。如果 $z$ 是高维变量，而且 $p_\theta(x\mid z)$ 由非线性神经网络定义，这个积分通常没有简单的闭式解。

未归一化的后验仍然可以计算：

$$
p_\theta(z\mid x)\propto p_\theta(x\mid z)p(z).
$$

我们可以为一个给定的 $z$ 计算右侧，但很难直接从归一化后的后验得到相互独立的样本。MCMC 不需要知道归一化常数也可以采样，但速度可能较慢，得到的样本之间也可能存在相关性。面对多峰后验时，MCMC 还可能难以在不同模式之间移动。

## 变分推断

变分推断使用容易处理的分布 $q_\phi(z\mid x)$ 近似困难的后验。我们先选择一个分布族 $\mathcal{Q}$，然后求解

$$
q_\phi^*(z\mid x)
=\arg\min_{q_\phi\in\mathcal{Q}}
D_{\mathrm{KL}}\left(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
\right).
$$

这是一个 Reverse KL 目标。期望由 $q_\phi$ 加权，而它正是我们知道如何采样的分布。关于 KL 方向的区别，可以参考 [[notes/kl-divergence|KL 散度]]。

## 从 Reverse KL 到 ELBO

直接写出后验似乎仍然需要未知的证据。展开 KL 以后，可以看到为什么这不是问题：

$$
\begin{aligned}
D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p_\theta(z\mid x)\right)
&=\mathbb{E}_{q_\phi}\left[
\log q_\phi(z\mid x)-\log p_\theta(x,z)
\right]
+\log p_\theta(x)\\
&=\log p_\theta(x)-\mathcal{L}_{\mathrm{ELBO}}(x).
\end{aligned}
$$

证据 $\log p_\theta(x)$ 不依赖变分参数 $\phi$。因此，最小化 KL 等价于最大化

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=\mathbb{E}_{q_\phi(z\mid x)}\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

它被称为下界，因为

$$
\log p_\theta(x)
=\mathcal{L}_{\mathrm{ELBO}}(x)
+D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p_\theta(z\mid x)\right)
\geq \mathcal{L}_{\mathrm{ELBO}}(x).
$$

## 与变分自编码器的关系

在 VAE 中，encoder 会产生一个容易处理的近似后验。常见选择是

$$
q_\phi(z\mid x)
=\mathcal{N}\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

Decoder 定义似然 $p_\theta(x\mid z)$。ELBO 可以写成

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p(z)\right).
$$

第一项要求采样得到的潜变量能够解释观测。第二项让近似后验保持接近先验。

## 近似可能遗漏什么

变分推断把推断问题转化为优化问题，通常比为每个观测运行新的采样过程更快。最终结果会受到分布族 $\mathcal{Q}$ 的限制。简单的高斯分布无法表示所有后验形状。当真实后验有多个模式时，Reverse KL 还可能只关注其中一个模式，并低估不确定性。

## 后续问题

- 什么是 mean-field 变分推断？
- 变分推断与 MCMC 有什么区别？
- VAE 中的 amortized inference 增加了什么？
- Reparameterization trick 为什么可以用于梯度训练？

## 参考资料

- [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
