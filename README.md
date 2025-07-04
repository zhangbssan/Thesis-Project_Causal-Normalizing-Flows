# 🎓 Graduation Thesis: Causal Modeling with Structural Causal Models (SCMs)

Welcome! This repository contains the code, experiments from my graduation thesis, which explores how **Structural Causal Models (SCMs)** can be used to model and analyze data-generating processes — both in theory and in practice.

## 🧠 Project Overview

This project investigates **causal machine learning** through the lens of SCMs. It operates under the assumption that features in the dataset are **causally related** via underlying functional dependencies.

The project is driven by two core goals:

1. **Learning the data-generating process** using machine learning models based on the causal structure, which is represented by Causal Graph(DAG).
2. **Analyzing the effects of structural mismatches**, especially how incorrect assumptions about causality impact **counterfactual reasoning**.

Modeling counterfactuals in a **trustworthy** way — such that their distributions remain valid — is a core challenge in this work.

---

## 🧮 Linear Case

In the linear scenario, I use a **closed-form solution** via **maximum likelihood estimation (MLE)** to directly model the data-generating process.

### Techniques

- Analytical modeling using **linear algebra** and **matrix operations**
- Code implementation in **NumPy**
- **Unit tests** written with **Pytest** to validate correctness

---

## 🔀 Nonlinear Case

For nonlinear dependencies, I implemented a **Causal Normalizing Flow (CNF)** — an **autoregressive model** that learns complex data-generation processes, assuming the causal graph is respected (via **masked layers by adjacency matrix**).

### Key Contributions

- Full implementation of a **CNF model** using **PyTorch**
- Extensions to test what happens when the **causal graph is mismatched**:
  - Designed new algorithms to compute **counterfactuals** under structural errors
  - Introduced **quantitative metrics** to measure distributional shifts
  - Built **visualizations** using **Seaborn** and **Matplotlib,** for interpretability

---

## 📊 Highlights

- 📚 Deep dive into **causal theory** and **SCM-based modeling**
- 🔍 Introspective experiments to evaluate **model reliability** under incorrect causal assumptions
- 🧪 Reproducible pipelines for **both linear and nonlinear** generative processes
- 📈 Clear and interpretable **visual outputs** to support findings

---

## 🔧 Tech Stack

- **Languages**: Python
- **Libraries**: NumPy, PyTorch, Pandas, Seaborn, Pytest
- **Concepts**: SCMs, MLE, CNF, Counterfactual Inference

---


