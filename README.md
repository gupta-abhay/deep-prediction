# Deep Prediction for Self-Driving Cars

This repository contains the most recent work for the MSCV Capstone Project
[Deep Prediction for Self-Driving Cars](https://mscvprojects.ri.cmu.edu/2019teamg/). We explore several different models and optimizers on the recently popular [Argoverse](https://www.argoverse.org/) data.

## Installation

Most of the work is done on Pytorch 1.2.0 (latest stable release, effective 09/19/2019) and using the [argoverse-api](https://github.com/argoai/argoverse-api).

To replicate the environment, run

```shell
conda env create -f environment.yml
```

This creates the environment, which can be started using,
```shell
conda activate deep_predict_argo
```

Then to install the argoverse-api, follow the instructions given [here](https://github.com/argoai/argoverse-api#installation). We have added our own code to api to enable several missing features for our modelling.

## Argoverse Extra Features

> Nitin to fill this in .....

## Models

- LSTM Baseline (w/ XY and Centerline data)
- Social-LSTM (with universal pooling module)
- TCN Baseline
- TrellisNet Baseline
- Transformer DEQ Baseline
- Stochastic-TCN (in development)


## Execution

To train the models, run 

```
python train_modified.py --model <model_name> --mode 'train'
```

To train the models, run 

```
python train_modified.py --model <model_name> --mode 'validate'
```

Look at `train_modified.py` on how to pass model_name argument.


## Optimizers

These are some recently introduced optimizers introduced (NIPS / ICCV / ICLR 2019). We have implemented the code for them but have not used them in the modelling currently.

- LookAhead Optimizer
- RectifiedAdam Optimizer [being used for TrellisNet and Transformer training]
- Ranger (RAdam + LookAhead) 
- Ralamb (RAdam + LARS + LookAhead)