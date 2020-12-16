
#%%
import pandas as pd

#to plot within notebook
import seaborn as sns 
import matplotlib.pyplot as plt


#%%
class DiscovData:

    def __init__(self, df): 
        self.df = df

    def targDistribut(self, y_train):
        return y_train.value_counts()

    def sentimentPlolt(self, y_train,y_test):
        #sns.countplot(self.df.Sentiment)
        sentimentTotal = [y_train,y_test]

        fig,ax = plt.subplots(1,2,figsize=(20,5))
        for i,data in enumerate(sentimentTotal):
            sns.countplot(data,ax=ax[i])


    def isNan(self):
        return self.df.isna().sum()


    def firstObs(self):
        #print the head
        return self.df.head()


    def discoverData(self, X_train):
        lst = []
        for i in X_train:
            lst.append(len(i))
    
    
        len1=pd.DataFrame(lst)
        return len1.describe()



    

    



# %%
