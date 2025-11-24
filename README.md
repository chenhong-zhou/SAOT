# SAOT (Spectral Attention Operator Transformer)
This repository contains the official implementation for "SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs" (AAAI 2026).


## Usage

To reproduce the results on the six operator learning benchmarks (Darcy, Navier-Stokes, Airfoil, Pipe, Plasticity, Elasticity), please follow these steps:

1. Download the datasets and update the data path.

2. Execute the corresponding shell script in the ./scripts directory to initiate training. For example

```
bash ./scripts/bash_darcy.sh
```


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{zhou2025dual,
  title={SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs},
  author={Chenhong Zhou and Jie Chen and Zaifeng Yang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/Geo-FNO

https://github.com/thuml/Transolver

https://github.com/idiap/fast-transformers/tree/master

https://github.com/YehLi/ImageNetModel

https://github.com/vivekoommen/NeuralOperator_DiffusionModel/tree/main


