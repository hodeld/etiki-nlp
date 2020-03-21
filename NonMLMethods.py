import numpy as np

def CreateCountVectorizer():
  from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
  import sklearn
  import nltk
  nltk.download('punkt')
  vect = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize,stop_words='english')
  return vect

def TrainLogisticRegression(train_df, vect):
  from sklearn.model_selection import GridSearchCV
  from sklearn.linear_model import LogisticRegression
  param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
  
  trainReg = vect.fit_transform(train_df["text"])
  
  grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
  grid.fit(trainReg, train_df['label'])
  feature_names = vect.get_feature_names()
  print("Best cross-validation score: {:.2f}".format(grid.best_score_))
  print("Best parameters: ", grid.best_params_)
  print("Best estimator: ", grid.best_estimator_)

  return grid


def EvalLogisticRegression(gridSearchResult, train_df, eval_df, vect):    
  trainReg = vect.fit_transform(train_df["text"])
  testReg = vect.transform(eval_df["text"])

  lr = gridSearchResult.best_estimator_
  lr.fit(trainReg, train_df["label"])

  evalPred = lr.predict(testReg)

  return evalPred
   
def TrainNaiveBayes(train_df, vect):
  nltk.download('punkt')

  vectTrainData = vect.fit_transform(train_df["text"])

  return vectTrainData


def EvalNaiveBayes(vect, model, train_df, eval_df):
  vectEvalData = vect.transform(eval_df["text"])

  from sklearn.naive_bayes import MultinomialNB
  clf = MultinomialNB()
  clf.fit(model, train_df["label"])

  evalPred = clf.predict(vectEvalData)

  return evalPred

def TrainAfinn(data):
  from afinn import Afinn
  af = Afinn()

  # compute sentiment scores (polarity) and labels
  sentiment_scores = [af.score(article) for article in data['text']]

  sentiment_category = [2 if score > 8 
                            else 0 if score < -8 
                                else 1 
                                for score in sentiment_scores]
  return sentiment_category                       

def LogisticRegressionMulti(train, test):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.preprocessing import MultiLabelBinarizer
  from sklearn.linear_model import LogisticRegression
  from sklearn.multiclass import OneVsRestClassifier
  vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,2),max_df=0.9,min_df=5,token_pattern=r'(\S+)')
  x_train = vectorizer.fit_transform(train[:,2])
  x_test = vectorizer.transform(test[:,2])
  y_train = train[:,1]
  y_test = test[:,1]
  
  mlb = MultiLabelBinarizer()
  label_train = mlb.fit_transform(y_train)
  label_test = mlb.fit_transform(y_test)
  
  model = LogisticRegression(C=1.0, penalty='l1', dual=False, solver='liblinear')
  model = OneVsRestClassifier(model)
  model.fit(x_train, label_train)
  pred = model.predict_proba(x_test)
  return pred, label_test


