---
title: KL Divergence
description: Notes on KL divergence in the context of variational autoencoders.
tags:
  - ml/information-theory
  - dl/vae
aliases:
  - en/concepts/kl-divergence
lang: en-US
status: seedling
publish: true
created: 2026-08-28
---

# KL Divergence

## Definition and notation

Let $P$ be the target distribution and $Q$ be another distribution used to approximate it. For discrete random variables, the KL divergence is

$$
D_{\mathrm{KL}}(P \parallel Q)
= \sum_x p(x) \log \frac{p(x)}{q(x)}
= \mathbb{E}_{x \sim P}\left[\log p(x)-\log q(x)\right].
$$

For continuous random variables, the sum becomes an integral:

$$
D_{\mathrm{KL}}(P \parallel Q)
= \int p(x) \log \frac{p(x)}{q(x)}\,dx.
$$

The expectation is taken under $P$. This means that regions with high probability under $P$ receive more weight. If $p(x)>0$ but $q(x)=0$, the KL divergence is infinite.

KL divergence is always nonnegative, and it is zero only when $P$ and $Q$ are the same almost everywhere. It is not symmetric, so $D_{\mathrm{KL}}(P \parallel Q)$ and $D_{\mathrm{KL}}(Q \parallel P)$ usually have different values.

## Relationship to cross-entropy

The cross-entropy between $P$ and $Q$ is

$$
H(P,Q)=-\sum_x p(x)\log q(x),
$$

while the entropy of $P$ is

$$
H(P)=-\sum_x p(x)\log p(x).
$$

These quantities are related by

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P \parallel Q).
$$

When $P$ is the fixed target distribution, $H(P)$ is constant. Changing $Q$ only changes the KL divergence term. Therefore, minimizing cross-entropy with respect to $Q$ is equivalent to minimizing $D_{\mathrm{KL}}(P \parallel Q)$. If the model can represent $P$, the minimum is reached when $Q=P$. If it cannot, training selects the available $Q$ with the smallest KL divergence from $P$.

## Relationship to likelihood

Suppose the data follow an unknown distribution $P_{\mathrm{data}}$, and a model assigns probability or density $q_\theta(x)$. For observed data $x_1,\ldots,x_N$, the likelihood is

$$
L(\theta)=\prod_{i=1}^{N}q_\theta(x_i).
$$

The data are fixed, so the likelihood is viewed as a function of the model parameters $\theta$. Maximum likelihood estimation is

$$
\theta_{\mathrm{MLE}}
=\arg\max_\theta \sum_{i=1}^{N}\log q_\theta(x_i)
=\arg\min_\theta \left[-\frac{1}{N}\sum_{i=1}^{N}\log q_\theta(x_i)\right].
$$

The expression on the right is the empirical negative log-likelihood. It estimates the population cross-entropy

$$
\mathbb{E}_{x\sim P_{\mathrm{data}}}[-\log q_\theta(x)]
=H(P_{\mathrm{data}},Q_\theta)
=H(P_{\mathrm{data}})+D_{\mathrm{KL}}(P_{\mathrm{data}}\parallel Q_\theta).
$$

Because $P_{\mathrm{data}}$ is fixed, its entropy does not depend on $\theta$. In expectation, maximizing likelihood is therefore equivalent to minimizing the forward KL divergence from the data distribution to the model distribution.

## Common applications

### Variational autoencoders

A VAE maximizes the evidence lower bound

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p(z)\right).
$$

The first term rewards reconstruction likelihood. The KL term keeps the approximate posterior $q_\phi(z\mid x)$ close to the prior $p(z)$. This regularizes the latent space and makes sampling from the prior useful. The same objective also reduces the gap between the approximate posterior and the true posterior:

$$
\log p_\theta(x)-\mathcal{L}_{\mathrm{ELBO}}(x)
=D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p_\theta(z\mid x)\right).
$$

### PPO and RLHF

In RLHF, a KL penalty is often used to keep the learned policy $\pi_\theta$ close to a reference policy $\pi_{\mathrm{ref}}$:

$$
\max_\theta\;\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}\left(\pi_\theta(\cdot\mid x)\parallel\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

The reward encourages preferred outputs. The KL term limits how far the policy can move from the reference model. This helps preserve useful language behavior when the reward model is imperfect.

PPO itself usually controls policy updates with a clipped probability ratio between the new and old policies. This clipped objective is not the same as a KL penalty. PPO implementations may also monitor KL divergence or stop an update when it becomes too large. In RLHF, the additional KL term usually compares the learned policy with a fixed reference model.

### Knowledge distillation

Knowledge distillation trains a student distribution to match a teacher distribution. With temperature $\tau$, a common distillation loss is

$$
\mathcal{L}_{\mathrm{distill}}
=\tau^2 D_{\mathrm{KL}}\left(
p_{\mathrm{teacher}}^{(\tau)}(\cdot\mid x)
\parallel
q_{\mathrm{student}}^{(\tau)}(\cdot\mid x)
\right).
$$

The teacher distribution is fixed, so minimizing this KL divergence is equivalent to minimizing cross-entropy with the teacher's soft predictions as targets. A higher temperature exposes relative probabilities among classes. These soft targets can transfer more information than a single hard label.

## Questions to revisit

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
