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

## Forward and reverse KL

Let $P$ be the target distribution and $Q$ be the approximation. The two directions are

$$
\underbrace{D_{\mathrm{KL}}(P\parallel Q)}_{\text{forward KL}}
=\mathbb{E}_{x\sim P}\left[\log\frac{p(x)}{q(x)}\right],
$$

and

$$
\underbrace{D_{\mathrm{KL}}(Q\parallel P)}_{\text{reverse KL}}
=\mathbb{E}_{x\sim Q}\left[\log\frac{q(x)}{p(x)}\right].
$$

Inverting the ratio changes the sign inside the logarithm. If the expectation remained under $P$, the result would only be the negative of forward KL and would not be a divergence. Reverse KL also changes the weighting distribution from $P$ to $Q$. This combination produces its different behavior.

> [!important] Core intuition
>
> | Direction | Main penalty | Typical behavior |
> | --- | --- | --- |
> | Forward KL, $D_{\mathrm{KL}}(P\parallel Q)$ | **Missing regions that are real under $P$** | Mass-covering |
> | Reverse KL, $D_{\mathrm{KL}}(Q\parallel P)$ | **Assigning probability to regions that are unrealistic under $P$** | Mode-seeking |

For forward KL, a region where $p(x)$ is large but $q(x)$ is close to zero creates a large cost. Missing a mode of $P$ can make the divergence infinite. A region where $p(x)$ is small receives little weight, even if $q(x)$ is relatively large. This makes extra coverage cheap, but not completely free. Probability assigned there must come from somewhere else because $Q$ still has to sum or integrate to one.

For reverse KL, placing probability where $p(x)$ is close to zero creates a large cost. However, if $q(x)$ is close to zero in one of the modes of $P$, that missed mode receives little weight under $Q$. When $Q$ cannot represent every mode, it may prefer one high-density region and ignore the others.

This is why forward KL is often called mass-covering and reverse KL is often called mode-seeking. These labels describe common behavior when a restricted distribution $Q$ approximates a fixed target $P$. They are not universal guarantees for every optimization problem.

### Which direction do we usually optimize?

Forward KL is natural when we can draw samples from the target distribution $P$. It can be written as

$$
D_{\mathrm{KL}}(P\parallel Q)
=\mathbb{E}_{x\sim P}[\log p(x)]
-\mathbb{E}_{x\sim P}[\log q(x)].
$$

The first term does not depend on $Q$. Therefore, samples $x_i\sim P$ are enough to estimate the part that matters for optimization:

$$
-\mathbb{E}_{x\sim P}[\log q(x)]
\approx -\frac{1}{N}\sum_{i=1}^{N}\log q(x_i).
$$

This is why maximum likelihood and supervised cross-entropy usually have the forward direction. They strongly penalize the case where the data distribution has high probability but the model assigns low probability.

Reverse KL is natural in a different computational setting. Suppose we can sample from $Q$, while the target is known only through an unnormalized density

$$
p(x)=\frac{\widetilde{p}(x)}{Z}.
$$

Then

$$
D_{\mathrm{KL}}(Q\parallel P)
=\mathbb{E}_{x\sim Q}[\log q(x)-\log\widetilde{p}(x)]
+\log Z.
$$

The unknown normalizing constant $\log Z$ does not depend on $Q$. We can therefore optimize reverse KL by sampling from $Q$ without first solving the difficult problem of sampling from $P$.

> [!tip] Practical rule
> Use forward KL when target samples from $P$ are available. Use reverse KL when sampling from the approximation $Q$ is easy and the target $P$ can be evaluated up to a normalizing constant.

| Setting                                    | Common direction                                                               | Why this direction is practical                                                          |
| ------------------------------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Maximum likelihood and supervised learning | Forward, $D_{\mathrm{KL}}(P_{\mathrm{data}}\parallel Q_\theta)$                | The dataset provides samples from $P_{\mathrm{data}}$.                                   |
| Knowledge distillation                     | Forward, $D_{\mathrm{KL}}(P_{\mathrm{teacher}}\parallel Q_{\mathrm{student}})$ | The teacher provides the target probabilities.                                           |
| Variational inference                      | Reverse, $D_{\mathrm{KL}}(Q_\phi\parallel P_{\mathrm{posterior}})$             | We can sample from $Q_\phi$ and evaluate the posterior up to its normalizing constant.   |
| Variational autoencoders                   | Reverse, $D_{\mathrm{KL}}(q_\phi(z\mid x)\parallel p_\theta(z\mid x))$         | The ELBO avoids direct sampling from the intractable true posterior.                     |
| KL-regularized RL and RLHF                 | Reverse form, $D_{\mathrm{KL}}(\pi_\theta\parallel\pi_{\mathrm{ref}})$         | Samples come from the learned policy, while the reference model scores the same outputs. |

The most classical reverse-KL example is [[en/notes/bayesian-and-variational-inference|variational inference]]. A VAE is one instance of it. RLHF also commonly uses a reverse-form KL regularizer relative to a reference policy, although it is not trying to reproduce the reference exactly because reward optimization pulls the learned policy away from it. PPO itself should not be treated as another name for reverse KL.

### Mathematical properties

Both directions are nonnegative and equal zero only when the two distributions agree almost everywhere. Neither direction is a distance metric because KL is not symmetric and does not satisfy the triangle inequality.

KL divergence is jointly convex in its two distribution arguments. This does not mean that a neural network training objective is convex in its parameters, because the map from parameters to distributions can be nonconvex.

KL also satisfies the data processing inequality. Applying the same deterministic or stochastic transformation to both distributions cannot increase their KL divergence. Coarse observations can hide differences between distributions, but cannot create new ones.

For joint distributions, KL follows a chain rule:

$$
D_{\mathrm{KL}}(P(X,Y)\parallel Q(X,Y))
=D_{\mathrm{KL}}(P(X)\parallel Q(X))
+\mathbb{E}_{x\sim P(X)}\left[
D_{\mathrm{KL}}(P(Y\mid x)\parallel Q(Y\mid x))
\right].
$$

This property allows a sequence-level KL divergence to be decomposed into conditional terms. It is useful for autoregressive models and policies.

Finally, forward and reverse KL become locally similar when $P$ and $Q$ are already close. If $q(x)=p(x)+\varepsilon(x)$, $\int\varepsilon(x)\,dx=0$, and $\varepsilon$ is small compared with $p$, then both directions have the same second-order approximation:

$$
D_{\mathrm{KL}}(P\parallel Q)
\approx D_{\mathrm{KL}}(Q\parallel P)
\approx \frac{1}{2}\int\frac{\varepsilon(x)^2}{p(x)}\,dx.
$$

Their different behavior becomes important when the distributions are far apart, have different support, or contain several modes.

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
- [Lecture Notes on Statistics and Information Theory](https://web.stanford.edu/class/stats311/lecture-notes.pdf)
- [Challenges and Opportunities in High-dimensional Variational Inference](https://arxiv.org/abs/2103.01085)
