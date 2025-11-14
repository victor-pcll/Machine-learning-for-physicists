# PHYS-467 — Machine Learning for Physicists  
## Graded Homework 1 — 2025  
### Student: Peucelle Victor

---

## 📌 Overview

This repository contains my solutions for **Homework 1** of the course **PHYS-467 — Machine Learning for Physicists**.  
The homework focuses on **robust linear regression**, **Huber loss**, **regularization techniques**, and **Monte-Carlo sampling** using the **Metropolis–Hastings algorithm**.  
A final part explores **real-world data** affected by outliers.

All results were obtained following the course instructions, with special care toward reproducibility, code clarity, and compliance with allowed libraries.

---

## ⚠️ Homework Rules (Important)

### ▸ Allowed libraries
- **Exercises 1 and 2:**  
  ✔️ `numpy`  
  ✔️ `matplotlib`  
  ❌ *No other packages allowed*  
- **Exercise 3:**  
  ✔️ `numpy`  
  ✔️ `matplotlib`  
  ✔️ `scikit-learn`

### ▸ Reproducibility
Randomness plays a role in all exercises, so:
- All random seeds are fixed using `np.random.seed(...)`
- Results may show small fluctuations due to finite-size effects

### ▸ Execution integrity
Before submission:
- All notebook cells run from top to bottom without any errors
- All plots and results regenerate correctly

---

# 🧮 Exercise 1 — Robust Regression with Huber Loss (11 pts)

This part investigates **linear regression** in the presence of **outliers**, and compares:
- **MSE loss**  
- **Huber loss** (scale parameter \( a \))  
- **L2 (ridge)** regularization  
- **L1 (lasso)** regularization  

The model follows:

$$
\hat{y}_\mu(w; x_\mu)=\frac{x_\mu^\top w}{\sqrt{d}}
$$

Residuals:

$$
r_\mu = y_\mu - \hat{y}_\mu
$$

Training objective:

$$
\mathcal L(w)=\frac{1}{n}\sum_{\mu=1}^n \ell(r_\mu)+ \lambda\,\mathcal R(w)
$$

where:
- $\ell(r)$: Huber loss  
- $\mathcal R(w)$: L2 or L1 regularization  

Goal:  
**Understand robustness to outliers and compare classical vs. robust losses.**

---

# 🎲 Exercise 2 — Monte-Carlo Sampling (8 pts)

### **Metropolis–Hastings Sampling from the Posterior**

Here, we compare:
- The **ERM estimator** using Huber loss  
- The **Bayes-optimal estimator**, computed via **MCMC**  

Data model:

$$
x_\mu \sim \mathcal N(0, I_d)
$$

Labels follow a contamination model:

$$
y_\mu =
\begin{cases}
  \dfrac{x_\mu^\top w^\star}{\sqrt d} + z_{\text{in}}, & z_{\text{in}} \sim \mathcal N(0, \Delta_{\text{in}}),\ \text{with prob. } 1-\varepsilon\\
  z_{\text{out}}\sim \mathcal N(m,\Delta_{\text{out}}), & \text{with prob. } \varepsilon
\end{cases}
$$

where $w^\star_i \in \{-1, +1\}$ independently.

Goal:  
**Compare Huber-based ERM to the Bayes estimator obtained from posterior sampling.**

---

# 📈 Exercise 3 — Real Sensor Data (4 pts)

You are given one-dimensional sensor measurements \(x_1\) and \(y\).  
During training, a system failure produced **outliers**.

Task:
- Fit **Linear Regression** (MSE)
- Fit **Huber Regression** (Huber loss)
- Compare robustness and prediction accuracy

Allowed tools:  
✔ `scikit-learn`  
❌ No manual gradient descent required

---

## 📂 Repository Structure

```
hw1/
│
├── README.md                # This file
├── homework1.ipynb          # Main notebook with complete solutions
├── data/                    # Real sensor dataset for Exercise 3
└── figures/                 # Generated plots (optional)
```

---

## 📦 Installation

To reproduce results:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🎯 Final Notes

This homework explores:

- Robust regression with Huber loss  
- High-dimensional inference under outliers  
- Bayesian estimation via MCMC  
- Real-world data analysis and model robustness  

The assignment combines theoretical understanding and practical implementation, providing deep insight into modern ML techniques relevant to physics.