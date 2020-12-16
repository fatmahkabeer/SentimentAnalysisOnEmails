
#%% import libraries
from DiscoverData import DiscovData
from DataPreparation import DataPreparation

from DLModeling import Modeling
from MLModeling import DecisionTreeModel
from MLModeling import LogisticRegressionModel

import pandas as pd # For DataFrame and handling

#to split the dataset into random train and test subsets
from sklearn.model_selection import train_test_split

#NLP
import nltk
#from textblob import TextBlob

#Keras
import keras
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

#Modeling
#from keras.models import Sequential
#from keras.layers import LSTM,Bidirectional,Dense,Embedding,Dropout
 


#%% Loading data
#https://www.kaggle.com/wcukierski/enron-email-dataset
emails = pd.read_csv('emails.csv', skiprows=lambda x:x%9)

#%% to see how emails look like
print(emails['message'][1])


#%% object of class DataPreparation
prep = DataPreparation(emails)

#to extract emails' body then add it to a new column
emails['Email'] = prep.bodyExtraction(emails['message'])

#labeling, adding label to each column.
emails['Sentiment'] = prep.labeling(emails['Email'])

#creating a separate dataset with just two columns body and sentiment
df = prep.newData(emails['Email'], emails['Sentiment'])

#%%
df.info()

#%%
#after extracting the body
print(df['Email'][1])


#%%split into train test sets
X = df['Email'] #Extracting data attributes
y = df['Sentiment'] # Extracting target/class labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# %% Discovering Data:
# Create an object of DiscovData class inside DiscoverData.py file.
dis = DiscovData(df)

# To figure out the target distribution
dis.targDistribut(y_train)

# to plot training and testing data 
dis.sentimentPlolt(y_train, y_test)

# to find missing values:
dis.isNan()

# print head
dis.firstObs()

#statistics:
dis.discoverData(X_train)


#%% Sentiment Analysis Preparing 
#Using The tokenizer Class to convert the sentences into word vectors

tokenizer=Tokenizer(199431,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(X_train)
len(tokenizer.word_index)

# training preparation:
Xtrain =  tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(Xtrain, maxlen=80,padding='post')
ytrain = y_train.replace({'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5, 'normal':6})
Ytrain= ytrain.values

#One hot Encoding the Emotion Values
Y_train_f=to_categorical(Ytrain) #Converts a class vector (integers) to binary class matrix.

# valedation preparation:
ytest= y_test.replace({'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5, 'normal':6})
X_val_f=tokenizer.texts_to_sequences(X_test)
X_val_pad=pad_sequences(X_val_f,maxlen=80,padding='post')
Y_val_f=to_categorical(ytest)



#%%
#Deep Learning
dModel = Modeling(df)
history = dModel.LSTM( X_train_pad,Y_train_f, X_val_pad,Y_val_f)

#%%
#Machain Learning
mModelDT = DecisionTreeModel(df)
mModelDT.model(X_train_pad, Y_train_f, X_val_pad, Y_val_f)

mModelLR = LogisticRegressionModel(df)
mModelLR.model(X_train_pad, ytrain, X_val_pad, ytest)


# %%
