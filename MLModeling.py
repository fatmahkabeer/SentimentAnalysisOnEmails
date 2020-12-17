#%%Importing required libraries

import pandas as pd

#Decision tree classifier from the sklearn library.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Logistic Regression from the sklearn library.
from sklearn.linear_model import LogisticRegression

#GridSearchCV
from sklearn.model_selection import GridSearchCV


#%% Modeling 
class DecisionTreeModel:

    def __init__(self, df):
        self.df = df

    def model(self, X_train_pad, Y_train_f, X_val_pad, Y_val_f):
        clf = DecisionTreeClassifier()
        
        #Training the decision tree classifier. 
        clf.fit(X_train_pad, Y_train_f)

        #Predicting labels on the test set.
        y_pred =  clf.predict(X_val_pad)

        x = ('Accuracy Score on train data: ', accuracy_score(y_true=Y_train_f, y_pred=clf.predict(X_train_pad)))
        y = ('Accuracy Score on test data: ', accuracy_score(y_true=Y_val_f, y_pred=y_pred))
        return (x, y)


    def modelHp(self, X_train_pad, Y_train_f, X_val_pad, Y_val_f):
        clf = DecisionTreeClassifier()
        
        param_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': range(20,30),
            #min_samples_leaf is also used to control over-fitting by defining that each leaf has more than one element.
            'min_samples_leaf': range(7,21)
        }

        grid = GridSearchCV(clf,
                            param_grid = param_dict,
                            cv = 10,
                            n_jobs = -1)

        #Training the decision tree classifier. 
        grid.fit(X_train_pad, Y_train_f)

        #Predicting labels on the test set.
        y_pred =  grid.predict(X_val_pad)

        x = ('Accuracy Score on train data: ', accuracy_score(y_true=Y_train_f, y_pred=grid.predict(X_train_pad)))
        y = ('Accuracy Score on test data: ', accuracy_score(y_true=Y_val_f, y_pred=y_pred))
        return (x, y)

       

class LogisticRegressionModel:

    def __init__(self, df):
        self.df = df

    def model(self, X_train_pad, ytrain, X_val_pad, ytest):
        clf = LogisticRegression()

        #Training the logistic regression classifier. 
        clf.fit(X_train_pad, ytrain)

        #Predicting labels on the test set.
        y_pred =  clf.predict(X_val_pad)

        x = ('Accuracy Score on train data: ', accuracy_score(y_true=ytrain, y_pred=clf.predict(X_train_pad)))
        y = ('Accuracy Score on test data: ', accuracy_score(y_true=ytest, y_pred=y_pred))
        return (x, y)

    
    def modelHp(self, X_train_pad, ytrain, X_val_pad, ytest):
        clf = LogisticRegression()
        
        param_dict = {
            #A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression
            'penalty': ['l1', 'l2'],
            #use paramter C as our regularization parameter.Parameter C = 1/λ
            #Lambda (λ) controls the trade-off between allowing the model to increase it's complexity as much as it wants with trying to keep it simple.
            'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]
        }

        grid = GridSearchCV(clf,
                            param_grid = param_dict,
                            cv = 10,
                            n_jobs = -1)

        #Training the decision tree classifier. 
        grid.fit(X_train_pad, ytrain)

        #Predicting labels on the test set.
        y_pred =  grid.predict(X_val_pad)

        x = ('Accuracy Score on train data: ', accuracy_score(y_true=ytrain, y_pred=grid.predict(X_train_pad)))
        y = ('Accuracy Score on test data: ', accuracy_score(y_true=ytest, y_pred=y_pred))
        return (x, y)
        #return grid.best_score_




# %%
 
