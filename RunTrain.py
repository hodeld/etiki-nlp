#!/usr/bin/python
# -*- coding: utf-8 -*-
# This file has the requirement that every library is already installed.
# This is a example implementation for a multi-label classification model and its training.
def trainTheModel():
    import numpy as np
    import pandas as pd
    import torch
    import copy
    import sklearn
    import csv
    
    # CSVReader is optional, we recommend using a database
    
    from CsvReader import ReadCsv
    from HelperFunctions import ShuffleData, ReplaceCategoriesWithIndex, OneHotEncodingForCategories, TransformDataIntoDataframe
    from TrainEvalModel import TrainModelForMultiLabel
    
    # We recommend replacing these two lines with database access and maybe feeding the data as a parameter
    
    folderPath = '/etiki-data'
    (data, categories, tendencies) = ReadCsv(folderPath, 'etikidata.csv', 'companies.csv', 'categories.csv', 'references.csv', 'tendencies.csv', 'topics.csv')
    
    #IMPORTANT! The input data in this method should be single-labeled, the following functions will convert them.
    rawData = data[:, [13, 4, 13]]
    
    multiLabelData = OneHotEncodingForCategories(ReplaceCategoriesWithIndex(categories, rawData, True))
    train_df = TransformDataIntoDataframe(multiLabelData)
    
    amount_of_categories = 5
    algo = 'roberta'
    model_name = 'roberta-base'
    
    # Change the output directory if you want to save the model else where.
    # These arguments are important for loading the model from disk. Make sure they are in a separate function, which the training and the prediction/loading can access.
    
    args = {
        'output_dir': 'outputs/',
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 6,
        'silent': True,
        'use_cached_eval_features': False,
        }
    model = TrainModelForMultiLabel(algo, model_name, train_df,
                                    amount_of_categories, args)
    
    			