# Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization

<img width="4013" height="767" alt="Image" src="https://github.com/user-attachments/assets/45fbb905-3a5d-4d02-b7e4-b969b8b072d9" />

## Overview

We propose **Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization (SGPC)**, a novel geometric deep learning framework that integrates sheaf theory, PAC-Bayesian regularization, and spectral optimization to achieve robust and generalizable node representations.

Our method introduces:
- Wasserstein–Entropic Sheaf Lifting for topological coupling between local feature spaces,  
- PAC-Bayes β–Dirichlet Calibration for uncertainty-aware generalization,  
- Spectral Gap Optimization for stability and expressiveness, and  
- SVR-AFM Layers for efficient semi-supervised learning on graph manifolds.

## Execution

To train SGPC on benchmark datasets (e.g., Cora, Citeseer, Pubmed, Chameleon, Squirrel, Actor, Cornell, Texas, Wisconsin) from 0 to 8:
- python main.py 0   # Cora
- python main.py 1   # Citeseer
- python main.py 2   # Pubmed
- ...
- python main.py 8   # Wisconsin

## Technical Appendix

Proofs of the theorems and technical details are provided in Supplemental.pdf
