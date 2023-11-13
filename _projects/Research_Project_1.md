---
title: 'Data-Driven Reachability Analysis'
date: 2023-11-12
permalink: /projects/data_driven_reachability_analysis
---
# Introduction

Our contributions are:
1. We propose a data-driven approach for fast convex hull approximation utilizing Input-Convex Neural Networks (ICNNs)
2. 

# Background

<details>
  <summary><h4>Input-Convex Neural Networks (ICNNs) and  </h4></summary>
</details>

<details>
  <summary><h4>Neural Network Representation of Discrete-Time Neural Network-Controlled Systems </h4></summary>
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
1. Given an initial set $$X$$, find an over-approximation of its image $$S(X)$$
2. Given a target set $$Y$$, find an over-approximation of its preimage $$S^{-1}(Y)$$

# Method
## Training

## Verification

# Results on Forward Reachability Analysis

# Currect and Future Work
    - Efficient Sample Strategy: this is important for preimage approximation. 
    - Disconnectivity of the preimage set: Due to
    - Verification on the preimage approximation



<details>
  <summary><h3>Hidden part: some discarded ideas and results</h3></summary>
</details>



