# MLP_NSGA: A Pytorch MLP Prediction with DEAP NSGAII Sample
By Yanren Qu and Ruijie Liu

## Introduction

This repo provides mlp pretraining model and training framework 
for architectural data with pytorch, 
as well as ONNX inference examples,
and a multi-objective optimization example with multiple constraints 
based on NsgaII algorithm with DEAP


PyTorch MLP pretrained model and training framework: 
This project provides a method for building and training MLP pretrained models using PyTorch. 
You can use this framework to train your architectural data model and track your trainning by tensorboard,
and use the pretrained model for prediction.

Training framework is built for general purpose, so you can simply 
replace backbone and loss function. 


ONNX inference examples: This project also provides example code for inference in the ONNX format. 
You can use this framework to deploy your pretrained model to production and perform inference in ONNX format.

Multi-objective optimization example based on Deap's MLP and NSGAII: 
This project also provides a multi-objective optimization example  with multiple constraints 
combined with our MLP model prediction 
based on Deap's NSGAII. 
You can use this example to explore how to optimize complex multi-objective problems using machine learning algorithms.

## Quick start

See sample_run.py


## Contact
```markdown
yanren@ualberta.ca
```
