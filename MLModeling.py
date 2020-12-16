#%%Importing required libraries

import pandas as pd

#Decision tree classifier from the sklearn library.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Logistic Regression from the sklearn library.
from sklearn.linear_model import LogisticRegression


#%% Modeling 
class DecisionTreeModel:

    def __init__(self, df):
        self.df = df

    def model(self, X_train_pad, Y_train_f, X_val_pad, Y_val_f):
        clf = DecisionTreeClassifier(criterion = 'entropy')
        
        #Training the decision tree classifier. 
        clf.fit(X_train_pad, Y_train_f)

        #Predicting labels on the test set.
        y_pred =  clf.predict(X_val_pad)

        x = ('Accuracy Score on train data: ', accuracy_score(y_true=Y_train_f, y_pred=clf.predict(X_train_pad)))
        y = ('Accuracy Score on test data: ', accuracy_score(y_true=Y_val_f, y_pred=y_pred))
        return (x, y)


class LogisticRegressionModel:

    def __init__(self, df):
        self.df = df

    def model(self, X_train_pad, ytrain, X_val_pad, ytest):
        logreg = LogisticRegression()

        #Training the logistic regression classifier. 
        logreg.fit(X_train_pad, ytrain)

        #Predicting labels on the test set.
        y_pred =  logreg.predict(X_val_pad)

        x = ('Accuracy Score on train data: ', accuracy_score(y_true=ytrain, y_pred=logreg.predict(X_train_pad)))
        y = ('Accuracy Score on test data: ', accuracy_score(y_true=ytest, y_pred=y_pred))
        return (x, y)




# %%
 
