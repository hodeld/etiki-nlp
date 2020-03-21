#!/usr/bin/python
# -*- coding: utf-8 -*-
# This file has the requirement that every library is already installed.
# This is an example implementation for loading a multi-label classification model from disk and predicting a text

def predictText(text):
    #Ideally the model wouldn't be loaded everytime someone predicts something, but kept in the GPU memory.
    import torch
    from TrainEvalModel import LoadMultiLabelModelFromDisk, PredictFromModel
    algorithm = 'roberta'
    modelFolder = 'outputs/'
    args = {
        'output_dir': modelFolder,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 6,
        'silent': True,
        'use_cached_eval_features': False,
    }
    model = LoadMultiLabelModelFromDisk(algorithm,modelFolder,args)
    prediction = PredictFromModel(model, text)

    #There are two options: 
    #   Either the threshold is set in here and prediction percentages are transformed to zeros and ones,
    #   or this could also be done outside of this method. In this file, we show how to transform it.

    threshold = 0.3
    result = [1 if predictionValue > threshold else 0 for predictionValue in prediction]
    return result

