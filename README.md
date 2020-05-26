# GraphLIME

GraphLIME is a model-agnostic, local, and nonlinear explanation method for GNN. It uses Hilbert-Schmit Independence Criterion (HSIC) Lasso, which is a nonlinear interpretable model. More details can be seen in the [paper](https://arxiv.org/pdf/2001.06216.pdf).

This repo implements GraphLIME by using awasome GNN library [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric), and reproduces the results of filtering useless features; that is, Figure 2 in the paper.

## Usage

All scripts of different experiments are in `scripts` folder. You can reproduce the results by:

```
sh scripts/noise_features_cora.sh
```

## Result

![](./images/cora.png)
![](./images/pubmed.png)

## Change Log

* 2020-05-26: first upload