# DIESGD
This repository contains the source code of DIESGD optimizer from our paper: Double-Integration-Enhanced Sochastic Gradient Descent Based on Neural Dynamics for Improving Generalization.

The core optimizer implementation code is placed under the directory "/optimizers".

## A quick look at the algorithm

![DIESGDOptimizer](DIESGDOptimizer.png)

## Useage in PyTorch

Simply put "/optimizers/DIESGD.py" in your main file path, and add this line in the head of your training script:

 `from DIESGD import DIESGD`

Change the optimizer as

`optimizer = DIESGD(model.parameters(), lr=0.01, lambda_1=1.0, lambda_2=1.0, gamma=1.0)`

Run your code.
