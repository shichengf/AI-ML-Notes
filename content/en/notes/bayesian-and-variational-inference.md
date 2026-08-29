---
title: Bayesian Inference and Variational Inference
description: A foundation for priors, likelihoods, posteriors, and variational approximations.
tags:
  - ml/bayesian-inference
  - ml/variational-inference
  - dl/vae
lang: en-US
status: seedling
publish: true
created: 2026-08-28
---

# Bayesian Inference and Variational Inference

## The basic problem

Bayesian inference is a framework for updating uncertainty after observing data. Let $x$ be the observed data and $z$ be an unknown quantity. Depending on the model, $z$ could be a parameter, a latent representation, or a hidden state.

> [!important] Core idea
> The prior describes what we believe about $z$ before observing $x$. The posterior describes what we believe after using the information in $x$.

## Prior, likelihood, evidence, and posterior

| Quantity                        | Meaning                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| Prior, $p(z)$                   | What the model assumes about $z$ before observing $x$.                                 |
| Likelihood, $p_\theta(x\mid z)$ | How likely the observed data are for a given value of $z$.                             |
| Evidence, $p_\theta(x)$         | The total probability or density assigned to $x$ after considering every possible $z$. |
| Posterior, $p_\theta(z\mid x)$  | The updated distribution of $z$ after observing $x$.                                   |

Bayes' rule connects them:

$$
p_\theta(z\mid x)
=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}.
$$

### How to read Bayes' rule

Read the two sides of the vertical bar first, then inspect the subscripts:

| Symbol | What it means |
| --- | --- |
| $x$ | The observed data. When computing the posterior, $x$ is treated as known. |
| $z$ | The unknown quantity to infer. It may be a parameter, latent variable, or hidden state. |
| $p(z)$ | The prior distribution of $z$. There is no conditioning bar because it describes our knowledge before observing $x$. |
| $p_\theta(x\mid z)$ | The likelihood of observing $x$ given $z$. The variable to the left of the bar is being described, while the expression to the right is given. |
| $p_\theta(z\mid x)$ | The posterior distribution of $z$ after observing $x$. Its conditioning direction differs from $p_\theta(x\mid z)$. |
| Subscript $\theta$ | Parameters that control the model distribution. The same $\theta$ means that the probabilities come from the same generative model. It is not another random variable. |
| $p_\theta(x)$ | The evidence, also called the marginal likelihood. It combines the explanations supplied by every possible $z$. |
| $\int \cdots dz$ | Integrate over every possible value of $z$. The final $dz$ identifies $z$ as the integration variable. A discrete model would use a sum over $z$. |
| $\propto$ | “Proportional to.” The equality omits a normalizing constant that does not change with $z$. Here, that constant is $p_\theta(x)$. |

> [!tip] Read it in one sentence
> After observing $x$, the posterior probability of a candidate $z$ depends on whether it was plausible beforehand and whether it can explain the current $x$. The denominator normalizes the results over all candidates.

The evidence is

$$
p_\theta(x)
=\int p_\theta(x\mid z)p(z)\,dz.
$$

The numerator asks whether a value of $z$ was plausible before seeing the data and whether it can explain the observed $x$. The denominator makes the posterior integrate to one.

## Why the posterior can be difficult

The main difficulty is often the evidence $p_\theta(x)$. Computing it requires integrating over every possible value of $z$. If $z$ is high-dimensional and $p_\theta(x\mid z)$ is defined by a nonlinear neural network, this integral usually has no simple closed form.

The unnormalized posterior is still available:

$$
p_\theta(z\mid x)\propto p_\theta(x\mid z)p(z).
$$

We can evaluate the right-hand side for a chosen $z$, but direct independent sampling from the normalized posterior is usually difficult. MCMC can sample without knowing the normalizing constant, but it may be slow and produce correlated samples. Multimodal posteriors can make mixing especially difficult.

## Variational inference

Variational inference replaces the difficult posterior with a tractable approximation $q_\phi(z\mid x)$. We first choose a family of distributions $\mathcal{Q}$ and then solve

$$
q_\phi^*(z\mid x)
=\arg\min_{q_\phi\in\mathcal{Q}}
D_{\mathrm{KL}}\left(
q_\phi(z\mid x)
\parallel
p_\theta(z\mid x)
\right).
$$

### How to read the variational inference objective

| Symbol | What it means |
| --- | --- |
| $p_\theta(z\mid x)$ | The true posterior that we want to approximate. $\theta$ contains the generative model parameters. |
| $q_\phi(z\mid x)$ | The distribution used to approximate the posterior. The letter $q$ marks it as a different distribution, and $\phi$ contains the approximation model parameters. |
| $\mathcal{Q}$ | The allowed family of approximate distributions, such as all Gaussian distributions with diagonal covariance. A calligraphic capital letter denotes a set of distributions. |
| $q_\phi\in\mathcal{Q}$ | The current approximation must belong to the selected family $\mathcal{Q}$. The symbol $\in$ means “belongs to.” |
| $\arg\min$ | Find the object that makes the following objective smallest. It returns the best $q_\phi$, not the minimum KL value. |
| Superscript $*$ | The best choice after optimization. Thus, $q_\phi^*$ is the best approximation available within the chosen family. |
| $\mathbb{E}_{q_\phi(z\mid x)}$ | Sample from $q_\phi(z\mid x)$ and take an average. The distribution under the expectation symbol supplies the weights. |
| $\mathcal{L}_{\mathrm{ELBO}}$ | $\mathcal{L}$ denotes the training objective. The subscript $\mathrm{ELBO}$ names it as the evidence lower bound. |

> [!tip] Read it in one sentence
> Search within the tractable family $\mathcal{Q}$ for a distribution $q_\phi$ that is as close as possible to the difficult posterior $p_\theta(z\mid x)$.

This is a reverse KL objective. The expectation is taken under $q_\phi$, which is the distribution we know how to sample from. For more about the direction of KL, see [[en/notes/kl-divergence|KL Divergence]].

## From reverse KL to the ELBO

Directly writing the posterior still appears to require the unknown evidence. Expanding the KL divergence shows why this is not a problem:

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

The evidence $\log p_\theta(x)$ does not depend on the variational parameters $\phi$. Minimizing the KL divergence is therefore equivalent to maximizing

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=\mathbb{E}_{q_\phi(z\mid x)}\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

It is called a lower bound because

$$
\log p_\theta(x)
=\mathcal{L}_{\mathrm{ELBO}}(x)
+D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p_\theta(z\mid x)\right)
\geq \mathcal{L}_{\mathrm{ELBO}}(x).
$$

## Connection to variational autoencoders

In a VAE, the encoder produces a tractable approximate posterior, often

$$
q_\phi(z\mid x)
=\mathcal{N}\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

The decoder defines the likelihood $p_\theta(x\mid z)$. The ELBO becomes

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\left(q_\phi(z\mid x)\parallel p(z)\right).
$$

The first term asks the sampled latent variable to explain the observation. The second term keeps the approximate posterior close to the prior.

## What the approximation can miss

Variational inference turns inference into optimization, which is often faster than running a new sampling procedure for every observation. The result depends on the chosen family $\mathcal{Q}$. A simple Gaussian approximation cannot represent every posterior shape. Reverse KL may also focus on one mode and underestimate uncertainty when the true posterior is multimodal.

## Questions to revisit

- What is mean-field variational inference?
- How is variational inference different from MCMC?
- What does amortized inference add in a VAE?
- Why does the reparameterization trick make gradient training possible?

## References

- [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
