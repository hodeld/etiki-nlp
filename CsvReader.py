def ReadCsv(path, dataCSV, companiesCSV, categoriesCSV, referencesCSV, tendenciesCSV, topicsCSV):
  import os
  import numpy as np
  import pandas as pd
  os.chdir(path)

  companies = pd.read_csv(companiesCSV,";")
  categories = pd.read_csv(categoriesCSV,";")
  references = pd.read_csv(referencesCSV,';')
  tendencies = pd.read_csv(tendenciesCSV,";")
  topics = pd.read_csv(topicsCSV,";")

  lineNum = 0
  headers = []
  articles = []
  continueAdding = False

  with open(dataCSV) as articleFile:
    lines = articleFile.readlines()    
    columnIndex = 0
    insertLine = []

    for line in lines:
      if lineNum == 0:
          # First Line has to include the headers to make sure the correct amount of colums are read
        headers = line.split('ÿ')      
        lineNum+=1
      else:
        splittedLine = line.split('ÿ')
        if len(splittedLine)==1: 
            #if the previous line in the CSV did not include a dividing character then add next line to it       
          insertLine[columnIndex-1] += splittedLine[0].rstrip()
          continueAdding = True
        else:
          for column in splittedLine:
              #if the previous line in the CSV did not include a dividing character then add to it until a dividing character shows up
            if continueAdding == True:
              insertLine[columnIndex-1] += column
              continueAdding = False
            else:
              columnIndex+=1
              insertLine.append(column.rstrip())
              #if there as many columns in the insertLine as there are headers columns, then add the insertLine to the articles
              if columnIndex == len(headers):
                columnIndex = 0
                articles.append(insertLine)
                insertLine = []            
          if len(insertLine) != 0:
            continueAdding = True

  realData = np.array(articles)
  return realData, categories, tendencies