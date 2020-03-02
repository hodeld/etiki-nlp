import random
import numpy as np
import pandas as pd

def ShuffleData(data):
  import random
  random.shuffle(data)
  return data

def replace_all(text, dic):
    for i, j in dic.items():
      text = text.replace(i, str(j))
    return text

def ReplaceSentimentsWithIndexes(data):
  #Data is expected to be [index, sentiment, text]
  dictionary = {'negative':'0','positive':'2','controversial':'1'}
  data[:,1]= [replace_all(p,dictionary) for p in data[:,1]]
  return data

def OneHotEncodingForCategories(data):
  from sklearn.preprocessing import MultiLabelBinarizer
  mlb = MultiLabelBinarizer()
  mlb.fit([['1','2','3','4','5']])
  transformedData = mlb.transform(data[:,1])

  for i, transformedEntry in enumerate(transformedData):
    data[:,1][i] = transformedEntry
  return data

def SplitArticlesIntoSentiments(data):
  positiveArticles = []
  negativeArticles = []
  controversialArticles = []
  for entry in data:
    if (entry[1]=='0'):
      negativeArticles.append(entry)
    elif (entry[1]=='1'):
      controversialArticles.append(entry)
    elif (entry[1]=='2'):
      positiveArticles.append(entry)

  positiveArticles = np.array(positiveArticles)
  negativeArticles = np.array(negativeArticles)
  controversialArticles = np.array(controversialArticles)
  return positiveArticles, negativeArticles, controversialArticles

def overSampleData(positiveArticles, controversialArticles, negativeArticles):
  positiveCount = positiveArticles[:,1].size
  controCount = controversialArticles[:,1].size
  negativeCount = negativeArticles[:,1].size

  if (positiveCount>controCount):
    i = 0
    while i+controCount<positiveCount:
      controversialArticles= np.vstack((controversialArticles,controversialArticles[i]))
      i+=1
    controCount += i
  elif (controCount>positiveCount):
    i = 0
    while i+positiveCount<controCount:
      positiveArticles= np.vstack((positiveArticles,positiveArticles[i]))
      i+=1
    positiveCount += i

  if (positiveCount>negativeCount):
    i = 0
    while i+negativeCount<positiveCount:
      negativeArticles = np.vstack((negativeArticles, negativeArticles[i]))
      i+=1
    negativeCount +=i
  elif (positiveCount<negativeCount):
    i = 0
    while i+positiveCount<negativeCount:
      positiveArticles = np.vstack((positiveArticles, positiveArticles[i]))
      i+=1
    positiveCount +=i

  if (negativeCount>controCount):
    i = 0
    while i+controCount<negativeCount:
      controversialArticles = np.vstack((controversialArticles, controversialArticles[i]))
      i+=1
    controCount += i
  elif (negativeCount<controCount):
    i = 0
    while i+negativeCount<controCount:
      negativeArticles = np.vstack((negativeArticles,negativeArticles[i]))
      i+=1
    negativeCount += i

  return positiveArticles,controversialArticles,negativeArticles

def underSampleData(positiveArticles, controversialArticles, negativeArticles):
  import statistics

  positiveCount = positiveArticles[:,1].size
  controCount = controversialArticles[:,1].size
  negativeCount = negativeArticles[:,1].size
  i = int(min(positiveCount,controCount,negativeCount)*0.7)
  trainData = np.vstack((positiveArticles[:i],controversialArticles[:i],negativeArticles[:i]))
  testData = np.vstack((positiveArticles[i:],controversialArticles[i:],negativeArticles[i:]))

  return trainData, testData

def TransformDataIntoDataframe(data):
  df={}

  df["text"]=[]
  df["label"]=[]
  for x in data:
    df["label"].append(x[1].astype(int))
    df["text"].append(x[2].replace('"',''))

  return pd.DataFrame.from_dict(df)  

def getMetricsMulti(cm, algorithm, folderName):
  import statistics
  import csv  
  print("Confusion Matrix:")
  print(cm)
  precision_arr = []
  recall_arr = []
  foneScore_arr = []
  precision_averages = []
  recall_averages = []
  foneScore_averages = []
  for i,matrix in enumerate(cm):
    TN = matrix[0][0]
    FN = matrix[0][1] 
    FP = matrix[1][0]
    TP = matrix[1][1]

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    foneScore = 2 * ((precision*recall)/(precision+recall))
    precision_arr.append(precision)
    recall_arr.append(recall)
    foneScore_arr.append(foneScore) 

    print ("Precision:",precision)
    print ("Recall:",recall)
    print ("F1-Score:",foneScore)

    with open('results/'+folderName+'/'+algorithm+'-class'+str(i)+'-precision.csv', 'a', newline='') as f:
      writer = csv.writer(f)        
      writer.writerow([precision])
    with open('results/'+folderName+'/'+algorithm+'-class'+str(i)+'-recall.csv', 'a', newline='') as f:
      writer = csv.writer(f)        
      writer.writerow([recall])
    with open('results/'+folderName+'/'+algorithm+'-class'+str(i)+'-foneScore.csv', 'a', newline='') as f:
      writer = csv.writer(f)        
      writer.writerow([foneScore])
  
  fields=[statistics.mean(precision_arr),statistics.mean(recall_arr),statistics.mean(foneScore_arr)] 

  with open('results/'+folderName+'/'+algorithm+'-average.csv', 'a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(fields)

def getMetrics(cm, labels, algoName):
  import statistics
  TP = np.diag(cm)
  FP = np.sum(cm, axis=0) - TP
  FN = np.sum(cm, axis=1) - TP
  TN = []
  for i in range(labels):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TN.append(sum(sum(temp)))
  
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  foneScore =  (2 *(precision*recall)/(precision+recall))
  
  avgPrecision = statistics.mean(precision)
  avgRecall = statistics.mean(recall)
  avgFoneScore = statistics.mean(foneScore)

  print ("Confusion Matrix:", cm)
  print ("Precision:",precision)
  print ("Recall:",recall)
  print ("F1-Score:",foneScore)
  print ("Avg Precision:", avgPrecision)
  print ("Avg Recall:", avgRecall)
  print ("Avg F1-Score:", avgFoneScore)

  import csv   
  fields=[avgPrecision,avgRecall,avgFoneScore]
  with open('results/'+algoName+'.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
  with open('results/'+algoName+'-precision.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(precision.tolist())
  with open('results/'+algoName+'-recall.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(recall.tolist())
  with open('results/'+algoName+'-foneScore.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(foneScore.tolist())

def CalculateWeights(labels, data):
  # data should be [id, label, text]
  from sklearn.utils.class_weight import compute_class_weight
  from scipy.special import softmax
  weights = compute_class_weight('balanced', labels, data[:,1])  
  weights = softmax(weights).tolist()
  return weights

def ReplaceCategoriesWithIndex(categories, data,multiIndex):    
  cateDictionary = categories["NAME"].T.to_dict()
  cateDictionary = {y:x for x,y in cateDictionary.items()}

  if (multiIndex):
    newData = []
    for entry in data:
      idx = -1
      for i,dataEntry in enumerate(newData):
        if (not isinstance(newData[i][1],list)):
          newData[i][1] = newData[i][1].split(",")
        if (dataEntry[0]==entry[0]):
          if (not entry[1] in newData[i][1]):        
            newData[i][1].append(entry[1])
          idx = i
          break
      if (idx == -1):
        newData.append(entry.tolist())
        
    return np.array(newData)
  data[:,1]= [replace_all(p,cateDictionary) for p in data[:,1]]
  return data

def ReplaceCategoriesWithIndexOneHot(categories, data):
  cateDictionary = categories["NAME"].T.to_dict()
  cateDictionary = {y:x for x,y in cateDictionary.items()}
  newData = []
  i = 0
  for i,entry in enumerate(data):
    newData.append(entry.tolist())
    if (not isinstance(newData[i][1],list)):
      newData[i][1] = newData[i][1].split(",")
    
  return np.array(newData)