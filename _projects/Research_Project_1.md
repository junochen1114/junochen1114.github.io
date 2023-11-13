---
title: 'Data-Driven Reachability Analysis / Neural Network Verification'
date: 2023-11-12
layout: archive
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

We further assume that both $$X$$ and $$Y$$ are convex polytopes.
The first problem is well-studied, and there are many existing methods to solve it, including 
[CROWN](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html) and [DeepPoly](https://dl.acm.org/doi/pdf/10.1145/3290354).
However, the second problem is much more challenging, especially when the activation function is not injective (e.g. ReLU).
To develop a general framework on both forward and backward directions, we propose a data-driven approach 
to solve the two problems. 

Given a sampled dataset $$Z$$ obtained from either $$S(X)$$ or $$S^{-1}(Y)$$, 
our approach mainly contains two steps:
1. Approximation: learn the convex hull shape by an ICNN
2. Verification: verify the approximation via an optimization problem

# Method
From the above background, an ICNN is a convex mapping with respect to its input. Moreover, an ICNN with 
Continuous Piecewise Affine (CPA) activation functions (e.g. ReLU, Leaky ReLU) is in essence a max-affine function.
Hence, it has the capability to approximate the convex hull of a dataset. This can be done by training the ICNN such that
the induced level-set polytope contains the dataset while the polytope volume is minimized.

$$
\begin{aligned}
\min_{\theta} &\quad \text{Vol}(P_{\theta}) \quad(\text{Volume Minimization Objective})\\
    \text{s.t.} &\quad Z \subseteq P_{\theta}  \quad (\text{Set Inclusion Constraint}) \\
                &\quad P_{\theta}  = {\{x \in \mathbb{R}^n | f_{\theta}(x) \leq 0\}}
\end{aligned}
$$

where $$f_{\theta}$$ is an ICNN with parameters $$\theta$$, and $$Z$$ is the sampled dataset.
Noting that without loss of generality, we fix the level set value to be $$0$$ since this is a just bias shift on ICNNs.

## Training
Given an ICNN, we train it such that the polytope boundaries match as closely as possible to the true
data convex hull, which is done by minimizing the following loss functions:

- Data Inclusion Loss: this part ensures that the polytope contains the dataset. The design of this loss term is intuitive,

- Volume Minimization Loss:

- Lipschitz Loss: this part is for ease of the following verification step by regularizing the Lipschitz constant of the ICNN.
By avoiding the dramatic change of neural network values, the aim of this loss term is to make the optimal value of the following verification optimization problem as small as possible.

## Verification
Given a well-trained ICNN $$f$$, we can verify its approximation by solving the following optimization problems:
### Forward Reachability

$$
\begin{aligned}
\max_{x} &\quad f(S(x)) \\
    \text{s.t.} &\quad x \in X
\end{aligned}
$$

It is notable that both $$f$$ and $$S$$ are neural networks, so is the composition. 
And the above optimization problem is essentially a neural network verification problem, where many frameworks provide accurate and efficient estimation on the optimal value.

### Backward Reachability
The backward reachability verification is more tricky, since it involves two neural networks, one occurs in the objective function and the
other occurs in the constraint.

$$
\begin{aligned}
\max_{x} &\quad f(x) \\
    \text{s.t.} &\quad AS(x) \leq b \quad (\text{our assumption on convex polytope}) \\
\end{aligned}
$$

Solving this optimization problem is in general intractable. Instead, we relax it.
The key idea is to replace $$S(x)$$ with some terms containing $$x$$, then the problem becomes a neural network verification problem again.

# Results on Forward Reachability Analysis

# Current and Future Work
- Efficient Sample Strategy: this is extremely important for preimage approximation. 
- Disconnectivity of the preimage set: Due to
- Verification on the preimage approximation



<details>
<summary><h1>Hidden part: some discarded ideas and results</h1></summary>
</details>


