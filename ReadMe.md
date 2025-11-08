# Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization

---

## 🧠 Overview

We propose **Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization (SGPC)** — a novel geometric deep learning framework that integrates sheaf theory, PAC-Bayesian regularization, and spectral optimization to achieve robust and generalizable node representations.

Our method introduces:
- **Wasserstein–Entropic Sheaf Lifting** for topological coupling between local feature spaces,  
- **PAC-Bayes β–Dirichlet Calibration** for uncertainty-aware generalization,  
- **Spectral Gap Optimization** for stability and expressiveness, and  
- **SVRG-GAT Layers** for efficient semi-supervised learning on graph manifolds.

---

To train SSGNN on benchmark datasets (e.g., Cora, Citeseer, Pubmed, Chameleon, Squirrel):
- python main.py --data 0   # Cora
- python main.py --data 1   # Citeseer
- python main.py --data 2   # Pubmed
- ...

