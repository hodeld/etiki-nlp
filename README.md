# etiki-nlp
This project contains the NLP-implementation of the bachelor-thesis "NLP für Ethik und Nachhaltigkeit". It was developed with Google Colab due to the graphics card demands of the NLP methods used and exported from there. This repository contains a sample implementation for a testing run of the sentiment analysis. 
## To be installed libraries by installpackages.sh
- Transformers by HuggingFace
- Seqeval
- Tensorboardx
- Torchvision
- Nvidia Apex
- SimpleTransformers

## Needed libraries outside of installpackages.sh
- numpy
- pandas
- random
- os
- sklearn.preprocessingsklearn.preprocessing.MultiLabelBinarizer
- statistics
- csv
- sklearn
- sklearn.utils.class_weight.compute_class_weight
- scipy.special.softmax

## Testing Run (sentiment analysis)
The testing run works as follows:
1. The code installs the needed packages with help of the file installpackages.sh. On Google Colab, the code inside of installpackages.sh needs to be included into the notebook and all lines need to be prefixed with an exclamation mark(!).
2. Next, the testing run reads the needed csv-files for the test run. This should be replaced with a database call to load the data.
3. The data then gets transformed according to the needs of simpletransformers. This means that the categories will be replaced with indizes starting with 0.
4. After that, the data gets split into positive, negative and controversial articles and each split dataset will be shuffled. 
5. The train and test data set will get assembled in a 70:30 split, ensure that this split will be used throughout all categories.
6. The selected transformer model can now be trained with the train data.
7. The trained model will be evaluated with the test data and all metrics will be written into .csv-files.
