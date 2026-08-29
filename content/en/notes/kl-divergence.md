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

### How to read this formula

Do not start by calculating it. First identify the role of each symbol:

| Symbol | What it means |
| --- | --- |
| $D_{\mathrm{KL}}$ | $D$ stands for divergence, a difference between two distributions. The subscript $\mathrm{KL}$ names the divergence. It is not another variable in the calculation. |
| $P\parallel Q$ | Read this as “$P$ relative to $Q$.” $P$ comes first and $Q$ comes second, so the order matters. $\parallel$ only separates the two distributions. It is not an absolute value or a conditional probability. |
| $P$ and $p(x)$ | Uppercase $P$ denotes the whole distribution. Lowercase $p(x)$ is the probability or probability density that $P$ assigns to one particular value $x$. The same relationship holds between $Q$ and $q(x)$. |
| $x$ | One possible value of the random variable. For a die roll, $x$ could be any number from 1 to 6. |
| $\sum_x$ | Go through every possible value of $x$ and add the terms. The $x$ under the summation sign tells us what is being summed over. |
| $\log\frac{p(x)}{q(x)}$ | Compare the probabilities that $P$ and $Q$ assign to the same $x$. This term is close to zero when the two probabilities are close. |
| $\mathbb{E}_{x\sim P}$ | Take a weighted average over all outcomes. The subscript $x\sim P$ says that $x$ follows $P$, so $P$ supplies the weights in the average. |
| $Q_\theta$ or $q_\phi$ | $\theta$ and $\phi$ denote model parameters that control a distribution. Training changes these parameters. |
| $P_{\mathrm{data}}$ | The subscript $\mathrm{data}$ labels the role of the distribution. It is not multiplication. Notation such as $P_{\mathrm{teacher}}$ works the same way. |
| $q_\phi(z\mid x)$ | The distribution of $z$, given $x$, is controlled by parameters $\phi$. The vertical bar $\mid$ means “given.” |

> [!tip] Read it in one sentence
> Let $x$ occur according to $P$, compare the probabilities that $P$ and $Q$ assign to the same $x$, and average those comparisons using $P$ as the weight.

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

Reverse KL is natural in a different computational setting. Suppose we can sample from $Q$, while the target is known only through an unnormalized density.

“Unnormalized” means that we can calculate a relative score $\widetilde p(x)$ for each candidate value, but we have not converted all scores into probabilities that sum to one.

> [!example] A simple example
> Suppose three candidate values have scores 2, 5, and 3. Dividing them by their total $Z=10$ gives the probabilities 0.2, 0.5, and 0.3. Even without knowing $Z$, we can still tell that the second candidate is more plausible than the first because $5>2$.

If there are millions of candidates, or if $x$ is a continuous high-dimensional variable, summing or integrating every score can be difficult. We may be able to evaluate $\widetilde p(x)$ for one chosen $x$ without being able to calculate the normalizing constant $Z$.

A Bayesian posterior is a common example. For a chosen latent value $z$, we can calculate

$$
\widetilde p(z)=p(x\mid z)p(z),
$$

but obtaining the posterior $p(z\mid x)$ requires summing or integrating over every possible $z$. That difficult total is the normalizing constant.

In general, the normalized target distribution can be written as

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

#### Why KL cannot be negative

The function $-\log t$ is convex. Assuming $q(x)>0$ wherever $p(x)>0$, Jensen's inequality gives

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=\mathbb{E}_{x\sim P}\left[-\log\frac{q(x)}{p(x)}\right] \\
&\geq -\log\mathbb{E}_{x\sim P}\left[\frac{q(x)}{p(x)}\right] \\
&= -\log\sum_{x:p(x)>0}q(x) \\
&\geq 0.
\end{aligned}
$$

The last step holds because the probability that $Q$ assigns to the support of $P$ cannot exceed one. Equality requires $q(x)/p(x)$ to be constant, and normalization then forces $P=Q$. If $p(x)>0$ somewhere that $q(x)=0$, the KL is positive infinity, so it is still not negative.

#### Why code can still report a negative value

| What produced the negative value? | What it means |
| --- | --- |
| A single sample or token | $\log p(x)-\log q(x)$ can be negative when $q(x)>p(x)$. Only its expectation under $P$ is KL. |
| A finite Monte Carlo average | The estimator has variance and may fall below zero even though its expectation is nonnegative. More samples should reduce this effect. |
| An unnormalized target | $\mathbb{E}_Q[\log q-\log\widetilde p]$ equals reverse KL minus $\log Z$. It is allowed to be negative when the missing constant is not restored. |
| An ELBO implementation | Training code often adds $-D_{\mathrm{KL}}$ to a maximization objective. The logged quantity may be negative KL rather than KL. |
| Floating-point error | A value that should be near zero can become slightly negative, especially with low precision or subtraction of similar log probabilities. |
| A density implementation error | Missing normalization, an omitted change-of-variables Jacobian, or inconsistent reduction can mean the computed quantity is not KL. |

When a negative value appears, first check whether the logged quantity is a complete, exact, normalized KL. A small negative Monte Carlo estimate can be legitimate, but a closed-form KL should normally remain nonnegative apart from tiny floating-point error.

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
