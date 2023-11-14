---
title: 'Data-Driven Reachability Analysis v'
date: 2023-11-12
layout: archive
permalink: /projects/data_driven_reachability_analysis
---
<div style="display: flex; flex-wrap: wrap; justify-content: flex-start;">
    <div style="margin: 10px; text-align: center;">
        <img src="../images/reach_intro/movie.gif" alt="Course Project 1" style="width: 40%; display: block; margin: auto;">
        <p>Step 1: Training Procedure</p>
    </div>
    <div style="margin: 10px; text-align: center;">
        <img src="../images/reach_intro/verify.png" alt="Course Project 2" style="width: 40%; display: block; margin: auto;">
        <p>Step 2: Verification</p>
    </div>
</div>

# Introduction
This project aims to 
Our contributions are:
1. We propose a data-driven approach for fast convex hull approximation utilizing Input-Convex Neural Networks (ICNNs)
2. 

# Background

<details>
<summary>

## Input-Convex Neural Networks (ICNNs)
</summary>
Our method heavily relies on DNs that fulfill specific constraints. These constraints 
result in DNs with a special property called input-convex, which is formulated as follows:
</details>

<details>
<summary>

## Neural Network Representation of Discrete-Time Neural Network-Controlled Systems
</summary>
text
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
Given an ICNN, we train it such that the polytope boundary matches as closely as possible to the true
data convex hull, which is done by minimizing the following loss functions:

### Data Inclusion Loss
This loss ensures that the polytope contains the dataset. The design of this loss term is intuitive,
with the general idea that all the data points should have function values less than or equal to zero.
Therefore, one possible choice for the loss term is 

$$ L(z; \theta) =  \text{sigmoid} (f_\theta(z)) $$

as $$L(z; \theta) \to 0$$ for every $$z \in Z$$, the approximated polytope $$P_{\theta}$$ contains the dataset $$Z$$.

### Volume Minimization Loss
This is the most challenging part of the training, since it is hard to directly express the volume of a polytope.
To overcome this difficulty, we introduce the distance of an inner point to the polytope boundary as

$$
\begin{aligned}
d(z, \partial P_\theta) = \max_{q} & \quad ||q-z||^2  \\
    \text{s.t.} &\quad f_\theta (q) \leq 0 \\
\end{aligned}
$$

and the volume of the polytope can be expressed as $$\sum_{z \in Z} d(z, \partial P_\theta)$$.

However, the above problem is still hard to solve, since it is a non-convex optimization problem. We propose several approximate forms that are
much more tractable, where the following method is the most efficient one:

Consider a given data point $$z \in Z$$ and unit direction $$v$$, the distance to the polytope boundary can be expressed as

$$
\begin{aligned}
d_v(z, \partial P_\theta) = \max_{\alpha} & \quad \alpha \\
    \text{s.t.} &\quad f_\theta (z+\alpha v) \leq 0 \\
                & \quad \alpha \geq 0
\end{aligned}
$$

Based on the convexity of $$f_\theta$$ and the constraints $$f_\theta (q) \leq 0$$, it's fair to assume that $$f_\theta (z+\alpha v)$$ is positive for large enough $$\alpha$$.
Thus, we apply a binary search on $$\alpha$$ to find the largest $$\alpha$$ with $$f_\theta (z+\alpha v) \leq 0$$.
It is notable that this method is gradient-based and would allow the parent optimization problem minimzing the sum of distances to propagate gradients to update $$\theta$$.
However, we experimentally find that this does not yield desireable results since the optimal $$\alpha$$ does not provide valuable gradient information for the parent optimization problem.

To overcome this, after finding the optimal $$\alpha$$, instead of minimizing these $$\alpha$$'s with respect to $$\theta$$ and applying backpropagation,
we maximize the function values at this point. The intuition behind this is that
if the boundary moves away from the data point, we should discourage this by enlarging the function value so that the zero level set squeezes towards the data point.

$$ L(z; \theta) =  -\text{sigmoid} (f_\theta(z+\alpha^* v))$$

### Lipschitz Loss
This part is for ease of the following verification step by regularizing the Lipschitz constant of the ICNN.
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
We perform our method on Double Integrator model. 
### Forward Reachability 
The baseline method is [AutomatedReach](https://proceedings.mlr.press/v211/entesari23a.html).
<table>
  <tr>
    <td style="padding-right: 10px;"><img src="../images/reach_horizon1/movie.gif" alt="Caption 1" width="100%"><br>Horizon 1: Training</td>
    <td style="padding-right: 10px;"><img src="../images/reach_horizon1/verify.png" alt="Caption 2" width="100%"><br>Horizon 1: Verification</td>
    <td><img src="../images/reach_horizon1/baseline.png" alt="Caption 3" width="100%"><br>Horizon 1: Baseline</td>
  </tr>
  <tr>
    <td style="padding-right: 10px;"><img src="../images/reach_horizon2/movie.gif" alt="Caption 4" width="100%"><br>Horizon 2: Training</td>
    <td style="padding-right: 10px;"><img src="../images/reach_horizon2/verify.png" alt="Caption 5" width="100%"><br>Horizon 2: Verification</td>
    <td><img src="../images/reach_horizon2/baseline.png" alt="Caption 6" width="100%"><br>Horizon 2: Baseline</td>
  </tr>
  <tr>
    <td style="padding-right: 10px;"><img src="../images/reach_horizon3/movie.gif" alt="Caption 7" width="100%"><br>Horizon 3: Training</td>
    <td style="padding-right: 10px;"><img src="../images/reach_horizon3/verify.png" alt="Caption 8" width="100%"><br>Horizon 3: Verification</td>
    <td><img src="../images/reach_horizon3/baseline.png" alt="Caption 9" width="100%"><br>Horizon 3: Baseline</td>
  </tr>
</table>



### Backward Reachability
Verification part is still under implementation yet. The baseline method is [INVPROP](https://arxiv.org/abs/2302.01404).
<table>
  <tr>
    <td>
      <img src="../images/backreach_horizon1/epoch60.png" alt="Caption 11" style="width: 45%; vertical-align: middle; margin-right: 10px;">
      <img src="../images/backreach_horizon1/step1_v.png" alt="Caption 12" style="width: 45%; vertical-align: middle;">
      <br>Horizon 1: Training | Horizon 1: Baseline
    </td>
  </tr>
  <tr>
    <td>
      <img src="../images/backreach_horizon2/epoch46.png" alt="Caption 4" style="width: 45%; vertical-align: middle; margin-right: 10px;">
      <img src="../images/backreach_horizon2/step2_v.png" alt="Caption 6" style="width: 45%; vertical-align: middle;">
      <br>Horizon 2: Training | Horizon 2: Baseline
    </td>
  </tr>
  <tr>
    <td>
      <img src="../images/backreach_horizon3/epoch70.png" alt="Caption 7" style="width: 45%; vertical-align: middle; margin-right: 10px;">
      <img src="../images/backreach_horizon3/step3_v.png" alt="Caption 9" style="width: 45%; vertical-align: middle;">
      <br>Horizon 3: Training | Horizon 3: Baseline
    </td>
  </tr>
</table>



# Current and Future Work
- Efficient Sample Strategy: this is extremely important for preimage approximation. Since we don't have a priori knowledge on the preimage set, 
we usually need to sample a large number of points to cover the preimage set. However, this is not efficient and we are working on this. An ideal
solution is to sample the points on the boundary of the image/preimage set, which is hard to obtain. Another solution is 
to make sure uniformly sampling points in the image/preimage set, which is also very challenging.

- Disconnectivity of the preimage set: due to the non-injectivity of the activation function, the preimage set can be disconnected.
Currect solution for other papers to reduce conservativeness is partitioning. However, this is not efficient under our framework since
we need to train a new ICNN for each partition.

- Verification on the preimage approximation: we relax the preimage approximation problem and hence introduce conservativeness.
Also, current verification methods are not exact. Since this part is not done yet, we are not sure how much conservativeness we introduce. 

<details>
<summary>

# Hidden part: some discarded ideas and results
</summary>
Our 
</details>


