# Machine Learning for Physicists — PHYS-467

## Overview

Machine learning and data analysis are becoming increasingly central to the scientific process, including in physics.  
This course introduces the fundamental concepts, algorithms, and tools of machine learning, with an emphasis on practical applications relevant to physicists.

The goal is to develop an intuition for key ML methods, understand their mathematical foundations, and learn how to implement them efficiently.

---

## Course Content

### 🔹 Introduction
- Examples of physics problems solvable with ML  
- Types of learning: supervised, unsupervised, and generative models  
- Overview of the ML pipeline: data → model → training → evaluation

### 🔹 Linear Regression
- Matrix formulation and least-squares estimation  
- Under-determined problems and ridge regularization  
- Polynomial regression  
- Bias–variance trade-off and overfitting  
- Train/validation/test splits

### 🔹 Probability Refresher & Bayesian Methods
- Key concepts from probability theory  
- Likelihood, MAP, and Bayesian inference  
- Least-squares as maximum likelihood with Gaussian noise  
- Regularization as prior information  
- Link with inverse problems and generalized linear models

### 🔹 Sparsity and Robustness
- Robust regression (Huber loss, LAD)  
- Sparse regression and LASSO  
- Role of sparsity in feature selection and compressed sensing

### 🔹 Optimization
- Gradient descent and stochastic gradient descent  
- Convergence principles and practical considerations

### 🔹 Classification
- Linear classifiers  
- Logistic regression and cross-entropy  
- Multi-class classification and one-hot encoding  
- K-nearest neighbours  
- Curse of dimensionality and geometric intuition

### 🔹 Unsupervised Learning & Dimensionality Reduction
- PCA and SVD  
- Low-rank approximation  
- Examples: recommender systems, geographical reconstruction from genomes, spin-glass card games

### 🔹 Statistical Mechanics & ML
- Analogy between inference and physics of high-dimensional systems  
- MAP ↔ ground-state search  
- MMSE ↔ thermal average  
- Bayesian inference ↔ Boltzmann sampling

### 🔹 Monte Carlo Methods
- Markov Chain Monte Carlo (MCMC)  
- Metropolis–Hastings algorithm  
- Gibbs sampling (heat bath)  
- Simulated annealing  
- EM algorithm and Bayesian hyper-parameter learning

### 🔹 Clustering
- K-means  
- Gaussian Mixture Models (GMMs)

### 🔹 Kernel Methods
- Non-linear regression via feature spaces  
- Representer theorem  
- Kernel ridge regression  
- SVM classification  
- Examples of kernels and their universality  
- Random feature regression & links to shallow neural networks

### 🔹 Neural Networks
- One-layer & multi-layer architectures  
- Universal approximation  
- Computational difficulties of training  
- Backpropagation and training with SGD  
- Hyperparameters: learning rate, batch size, initialization, etc.

### 🔹 Deep Learning
- Convolutional neural networks  
- Locality, weight sharing, translational symmetry  
- Over-parameterization and double descent  
- Implicit regularization  
- Transfer learning  
- Adversarial examples and data augmentation

### 🔹 Generative Models & Self-Supervised Learning
- Autoencoders  
- Boltzmann machines and maximum entropy principle  
- Flow-based and diffusion-based models  

### 🔹 Attention & Transformers
- Attention mechanism  
- Multi-head attention  
- Basics of transformer architectures  

---

## 🔧 Recommended Python Libraries

To run the notebooks and reproduce the examples, install the following libraries:

```
numpy  
scipy  
matplotlib  
pandas  
scikit-learn  
torch        # only needed for deep-learning sections  
jax          # optional, for autodiff exercises  
tqdm  
notebook or jupyterlab  
```

You can install everything using:

```bash
pip install numpy scipy matplotlib pandas scikit-learn torch jax tqdm notebook
```

---

## 🔍 Structure (Optional Suggestion)

You may structure your repository as follows:

```
/notebooks/      # Jupyter notebooks for each chapter  
/src/            # Helper functions and utilities  
/data/           # Datasets used during the course  
/figures/        # Saved plots and images  
README.md        # Course overview  
```