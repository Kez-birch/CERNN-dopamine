# CERNN 

This repository is the official implementation of [CERNN](coming soon). 

## Setting up the environment

## Requirements
- you may need to separately install pytorch with the right cuda setting [install pytorch](https://pytorch.org/get-started/locally/)
- training models requires an account on [weights and biases](wandb.ai)

Install the requirements with any package manager using the `requirements.txt.` 

```setup
pip install -r requirements.txt
```

## Training

To train the models, run this command:

CERNN: 
```train
python3 train.py 
```
Baseline LeakyRNN: 
```train
python3 train.py model=baseline_leaky_rnn
```

> Training uses [hydra](https://hydra.cc/docs/intro/), which lets you modify any hyperparameters and settings in the `src/hydraconfigs` folder via the command line


## Evaluation

Add a checkpoint folder with the following structure 
```
checkpoint_dir 
|- epoch_perf_.ckpt   # best performing model 
|- hp_pl_module.pkl   # model hyperparameters
|- last.ckpt          # final model after full training 
|- task_hp.pkl        # task configuration 
```

Add the path to the notebook in `notebooks/model_analysis.ipynb` for analysis


## Pre-trained Models

[UoB sharepoint link to models](https://uob-my.sharepoint.com/:f:/g/personal/od23963_bristol_ac_uk/EiDCEnjtVwFJrkKJ4nte-3YBTqjRt4f34bCwiP2hYF3jhQ?e=Y8e5cn)

## Results

Our model achieves the following performance on all tasks:


| Model name         | 100 epochs | 500 epochs
| ------------------ |---------------- | -------------- |
| CERNN defaults   |     93%        |            |
| CERNN xyz regulariser   |             |            |
| Baseline leaky RNN |             |            |




