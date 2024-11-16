# Methods with Local Steps and Random Reshuffling for Generally Smooth Non-Convex Federated Optimization

This repository provides the code to reproduce the results of the paper "Methods with Local Steps and Random Reshuffling for Generally Smooth Non-Convex Federated Optimization". 
It has three main scripts, which can be used to run 3 types of the experiments.

## `run_crr.py`
This script can be used to reproduce results for the first set of experiments for _Methods with random reshuffling subsection_ on 
$$f(x) = \frac{1}{N} \sum_{i=1}^{N} \|x - x_i\|^4,\ x_i \in [-10, 10]^d.$$

For information on arguments use `python run_crr.py --help`.

## `run_crr_torch.py`
This script can be used to reproduce results for the second set of experiments for _Methods with random reshuffling subsection_ on ResNet-18 on CIFAR10 dataset.

For information on arguments use `python run_crr.py --help`.

## `run_fl.py`
This script can be used to reproduce results for the third and fourth sets of experiments for _Methods with local steps_ and _Methods with local steps, random reshuffling and partial participation_ on $$f(x) = \frac{1}{N} \sum_{i=1}^{N} \|x - x_i\|^4,\ x_i \in [-10, 10]^d.$$
All the clients in these experiments are spawned due to technical limitations.

For information on arguments use `python run_fl.py --help`.
