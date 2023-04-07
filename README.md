# mdre
Estimating the Density Ratio between Distributions with High Discrepancy using Multinomial Logistic Regression

## Environment setup
We use two environments for our experiments. For all of the experiments except the high dimensional ones, we use PyTorch. For the high dimensional experiments, we use Tensorflow.

We provide conda environment configuration files to easy reproducibility: `torch_env.yml` and `tf_env.yml`

```
conda env create -f [***.yml]
```

## 1D Experiments
Use `notebooks/mdre-1d-exps.ipynb` in order to reproduce our MDRE 1D experiments on density ratio estimation. The default experiment and hyperparameters are set for the case in which `p~N(-1, 0.1)`, `q~N(1, 0.2)`, and `m~Cauchy(0, 1.0)`. The experiment configurations and hyperparameters can be changed in the third block of the notebook, if one would like to change the experiment settings.

## High-dimensional Experiments
Use `notebooks/mdre-highdim-exps.ipynb` in order to reproduce our MDRE high dimensional experiments on density ratio estimation / mutual information estimation. The default experiment and hyperparameters are set for the case in which `p~N(-1, 0.1)`, `q~N(1, 0.2)`, and `m~Cauchy(0, 1.0)`. The experiment configurations and hyperparameters can be changed in the third block of the notebook, if one would like to change the experiment settings.

## Omniglot experiments
### MDRE
Use `notebooks/omniglot-mdre.ipynb` in order to reproduce our MDRE representation learning experiments on SpatialMultiOmniglot. The default experiment is currently set to the one with 9 characters (3 x 3 grid of Omniglot characters) and the default hyperparameters are set for that experiment. If one would like to reproduce other experiments with 1 character or 4 characters (1 x 1 grid, 2 x 2 grid), then one only needs to change the according hyperparameters `NUM_CHARS` and `ALPHAS` mask size accordingly.

## BCDRE
Use `notebooks/omniglot-bcdre.ipynb` in order to reproduce BCDRE representation learning experiments on SpatialMultiOmniglot. The default is set to the one with 4 characters (2 x 2 grid of Omniglot characters). One can modify the same hyperparamters mentioned above to run this BCDRE baseline experiment for different configurations.

## Omniglot Data
Download the data using the following link and place it within `./data/omniglot` in this repository.

https://drive.google.com/file/d/1r5_r92wisYs4hSXk-jxdBVBuMHpwUbXX/view?usp=share_link
