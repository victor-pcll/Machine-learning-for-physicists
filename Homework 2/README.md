# 🔥 Forest Fire Dynamics with Transformers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Overview

This project explores the application of **Transformer architectures** to model and predict the dynamics of a forest fire simulation. The core objective is to predict the next state of a grid-based environment (Cellular Automata) where fire spreads according to specific stochastic rules.

A key focus of this study is the impact of **Positional Embeddings** on the model's ability to generalize, specifically when dealing with periodic boundary conditions (wrap-around grids).

## 🎯 Objectives

1.  **Model Dynamics:** Train a Transformer to learn the transition rules of a forest fire simulation (Empty $\to$ Tree $\to$ Fire).
2.  **Positional Encoding Analysis:** Compare different embedding strategies to handle the spatial geometry of the grid.
3.  **Topology Awareness:** Demonstrate why standard embeddings fail at boundaries in toroidal (periodic) environments and implement a solution.

## 🧪 Methodologies & Architecture

The project implements a Transformer-based neural network using **PyTorch**. The input is a flattened grid representing the current state of the forest, and the output is the predicted state of the grid at $t+1$.

### The Positional Embedding Study
Three distinct positional embedding approaches were implemented and compared:

* **1D Absolute Embedding:** Standard sequence positioning.
* **2D Absolute Embedding:** Grid-based positioning ($x, y$ coordinates).
* **2D Torus Embedding:** A specialized embedding mapping coordinates to a torus topology using sinusoidal functions. This accounts for the "wrap-around" nature of the simulation borders.

## 📊 Key Results

The experiments demonstrated that the **2D Torus Embedding** significantly outperformed standard approaches.

| Embedding Type | Observation |
| :--- | :--- |
| **Standard 2D** | Struggled at grid boundaries due to the discontinuity of coordinates, creating "edge artifacts" and higher error rates. |
| **2D Torus** | Successfully modeled the periodic boundary conditions, achieving the lowest Train/Test loss and eliminating boundary errors. |

> **Conclusion:** For environments with periodic boundary conditions (like many physics simulations or cellular automata), incorporating topological knowledge into the positional embedding is critical for model performance.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.x
* Jupyter Notebook
* PyTorch
* NumPy
* Matplotlib

### Running the Project

1.  Clone the repository:
    ```bash
    git clone [https://github.com/victor-pcll/Exploring-transformers-empirically](https://github.com/victor-pcll/Exploring-transformers-empirically)
    ```
2.  Install dependencies:
    ```bash
    pip install torch numpy matplotlib jupyter
    ```
3.  Open the notebook:
    ```bash
    jupyter notebook HW2_part1_of_2.ipynb
    ```
4.  Run all cells to train the models and generate the visualization plots.

## 📂 File Structure

* `HW2_part1_of_2.ipynb`: The main notebook containing data generation, model architecture, training loops, and evaluation/visualization.

## 🤝 Context

This project was developed as part of a Deep Learning coursework (HW2). It focuses on the practical implementation of attention mechanisms and the mathematical intuition behind positional encodings.

---

*Author: Peucelle Victor*