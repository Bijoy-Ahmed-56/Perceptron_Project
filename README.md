# Implementation and Comparison of Perceptron Training Methods

**Classic Perceptron and Gradient Descent**

This repository contains a complete university project implementing and comparing the classic Perceptron algorithm and various gradient-descent training methods from scratch using Python, NumPy, and Matplotlib (without using scikit-learn's Perceptron).

## Project Overview

This project uses a two-class, two-feature subset of the classic Iris dataset (`iris.csv`), specifically focusing on **Iris-setosa** and **Iris-versicolor** using **petal length** and **petal width** features.

### Key Components Implemented:

1. **Classic Perceptron**:
   - Implemented from scratch using NumPy in both **online** and **full-epoch** (batch-style accumulated) versions.
   - Tracks weight updates, weight vector evolution, and decision boundary snapshots[cite: 9].
   - Includes evaluation of convergence and margins.
2. **Gradient-Descent Perceptron**:
   - Implemented using the Perceptron criterion loss function[cite: 9].
   - Supports **Batch**, **Mini-batch**, and **Stochastic Gradient Descent (SGD)** training variants[cite: 9].
3. **Bonus Experiments**:
   - Evaluated the impact of different learning rates ($\eta$)[cite: 9].
   - Compared fixed presentation order versus shuffled order[cite: 9].
   - Tested zero initialization versus small random initialization[cite: 9].

---

## Project Structure

- `main.py` - Core implementation scripts for data loading, classic Perceptron, and gradient-descent models[cite: 9].
- `Perceptron_Full_Assignment_Report.docx` - Detailed formal project report[cite: 9].
- `iris.csv` - The filtered dataset containing Iris-setosa and Iris-versicolor samples[cite: 9].
- `*.png` - Generated plots illustrating weight evolutions, loss curves, and decision boundaries[cite: 9].

---

## Requirements

To run this project, make sure you have the following Python libraries installed:

```bash
pip install numpy matplotlib
```
