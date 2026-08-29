---
title: KL Divergence
description: The direction, optimization meaning, and useful behavior of KL divergence.
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

The first term does not depend on $Q$. Samples from $P$ are enough to estimate the second term. This is why maximum likelihood and supervised cross-entropy usually have the forward direction. They strongly penalize the case where the data distribution has high probability but the model assigns low probability.

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

For the mechanics of the classic reverse-KL case, see [[en/notes/bayesian-and-variational-inference|Bayesian Inference and Variational Inference]]. PPO itself should not be treated as another name for reverse KL.

### Useful properties

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

For observed samples $x_1,\ldots,x_N$, the model likelihood is

$$
L(\theta)=\prod_{i=1}^{N}q_\theta(x_i).
$$

Its average negative log-likelihood is the sample estimate of cross-entropy:

$$
-\frac{1}{N}\log L(\theta)
=-\frac{1}{N}\sum_{i=1}^{N}\log q_\theta(x_i).
$$

Maximizing likelihood, minimizing negative log-likelihood, minimizing cross-entropy, and minimizing forward KL are therefore the same optimization problem when the data distribution is fixed.

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Lecture Notes on Statistics and Information Theory](https://web.stanford.edu/class/stats311/lecture-notes.pdf)
- [Challenges and Opportunities in High-dimensional Variational Inference](https://arxiv.org/abs/2103.01085)
