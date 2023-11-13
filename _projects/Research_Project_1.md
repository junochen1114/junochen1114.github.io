---
title: 'Data-Driven Reachability Analysis'
date: 2023-11-12
permalink: /projects/data_driven_reachability_analysis
---
# Introduction
This projects
Our contributions are:
1. We propose a data-driven approach for fast convex hull approximation utilizing Input-Convex Neural Networks (ICNNs)
2. 

# Background

<details>
  <summary><h2>Input-Convex Neural Networks (ICNNs) and  </h2></summary>
  Our method heavily relies on DNs that fulfill specific constraints. These constraints 
  result in DNs with a special property called input-convex, which is formulated as follows:
</details>

<details>
  <summary><h2>Neural Network Representation of Discrete-Time Neural Network-Controlled Systems </h2></summary>
</details>


# Problem Formulation
Based on the above background, the closed-loop system can be exactly represented by a neural network,

$$
    x_{k+1} = f(x_k) 
$$

hence the state at time step $$k$$ can be represented by $$k$$-time composition of the neural network, as a new neural network, in terms of the initial state $$x_0$$:

$$
    x_k = f^k(x_0)
$$

This displays that multiple-step reachability analysis can be done starting at $$x_0$$ by multiple-step composition of the neural network, which is still a neural network.
So the reachability analysis problem is essentially computing the image or preimage of a set under some neural network.

From now on we assume that our system is a neural network, denoted as $$S$$, we address the following problems:
1. **(Forward Reachability)** Given an initial set $$X$$, find a tight over-approximation of its image $$S(X)$$
2. **(Backward Reachability)** Given a target set $$Y$$, find a tight over-approximation of its preimage $$S^{-1}(Y)$$

The first problem is well-studied, and there are many existing methods to solve it, including 
[CROWN](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html) and [DeepPoly](https://dl.acm.org/doi/pdf/10.1145/3290354)
However, the second problem is much more challenging, especially when the activation function is not injective (e.g. ReLU).
To develop a general framework on both forward and backward directions, we propose a data-driven approach 
to solve the two problems. Given a sampled dataset $$Z$$ obtained from either $$S(X)$$ or $$S^{-1}(Y)$$, 
our approach mainly contains two steps:
1. Approximation: learn the convex hull shape by an ICNN
2. Verification: verify the approximation via an optimization problem

# Method
From the above background, an ICNN is a convex mapping with respect to its input. Moreover, an ICNN with 
Continuous Piecewise Affine (CPA) activation functions (e.g. ReLU, Leaky ReLU) is in essence a max-affine function.
Hence, it has the capability to approximate the convex hull of a dataset. This can be done by training the ICNN such that
the induced level-set polytope contains the dataset while the polytope volume is minimized.

$$
\min_{\theta} \quad \text{Vol}(P_{\theta}) \\
    \text{s.t.} \quad \text{Conv}(X) \subseteq P_{\theta} \\
                \quad P_{\theta}  = {\{x \in \mathbb{R}^n | f_{\theta}(x) \leq 0\}}
$$
noting that without loss of generality, we fix the level set value to be 0 since this is a just bias shift on ICNNs.

## Training
Given an ICNN, we train it such that the polytope boundaries match as closely as possible to the true
data convex hull. 

## Verification
- Forward Reachability
- Backward Reachability

# Results on Forward Reachability Analysis

# Currect and Future Work
- Efficient Sample Strategy: this is important for preimage approximation. 
- Disconnectivity of the preimage set: Due to
- Verification on the preimage approximation



<details>
  <summary><h1>Hidden part: some discarded ideas and results</h1></summary>
</details>



