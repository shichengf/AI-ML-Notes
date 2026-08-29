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

The cross-entropy between (P) and (Q) is

$$
H(P,Q)=-\sum_x p(x)\log q(x),
$$

while the entropy of (P) is

$$
H(P)=-\sum_x p(x)\log p(x).
$$

These quantities are related by

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P \parallel Q).
$$

When $P$ is the fixed target distribution, $H(P)$ is constant. Changing $Q$ only changes the KL divergence term. Therefore, minimizing cross-entropy with respect to $Q$ is equivalent to minimizing $D_{\mathrm{KL}}(P \parallel Q)$. If the model can represent $P$, the minimum is reached when $Q=P$. If it cannot, training selects the available $Q$ with the smallest KL divergence from $P$.


## Why it appears in variational autoencoders



## Intuition

## Its role in the VAE objective

## Questions to revisit

## References
