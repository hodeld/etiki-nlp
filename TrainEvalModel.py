#----------------------------------------------------------------------------
# MULTI-CLASS / SENTIMENT ANALYSIS
def TrainModelForMultiClass(algorithm, base, training_df, num_labels, args, weights=None):
  from simpletransformers.classification import ClassificationModel

  # Create a TransformerModel
  model = ClassificationModel(algorithm, base, num_labels=num_labels, weight=weights, args=args)  
  
  model.train_model(training_df)
  return model

#----------------------------------------------------------------------------
#MULTI-LABEL / TEXT CLASSIFICATION
def TrainModelForMultiLabel(algorithm, base, training_df, num_labels, args, weight=None):
  from simpletransformers.classification import MultiLabelClassificationModel
  
  # Create a TransformerModel
  model = MultiLabelClassificationModel(algorithm, base, num_labels=num_labels, args=args, pos_weight=weight)

  model.train_model(training_df)
  return model

#----------------------------------------------------------------------------
#GENERAL METHODS
def EvalFromModel(model, evaluation_df):
  import sklearn
  # Evaluate the model
  result, model_outputs, wrong_predictions = model.eval_model(evaluation_df,acc=sklearn.metrics.accuracy_score,matr=sklearn.metrics.confusion_matrix)
  return result, model_outputs, wrong_predictions

def EvalFromMultiLabelModel(model, evaluation_df):
  result, model_outputs, wrong_predictions = model.eval_model(evaluation_df)
  return result, model_outputs, wrong_predictions

def PredictFromModel(model, text):
  predictions, raw_outputs = model.predict(text)
  return predictions, raw_outputs