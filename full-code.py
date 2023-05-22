# %%
# LOAD DATASET
from sklearn.datasets import load_iris
dataSet = load_iris()
features = dataSet.data
labels = dataSet.target                                      
labelsNames = list(dataSet.target_names)
featuresNames = dataSet.feature_names                         

print([labelsNames[i] for i in labels[47:52]])               
print(featuresNames)




# %%
# ANALAYZE DATA
import pandas as pd
print(type(features))
featuresDF= pd.DataFrame(features)
featuresDF.columns = featuresNames

print(type(featuresDF))
print(featuresDF.describe())                                 
print(featuresDF.info())                                     




# %%
# VISUALIZE DATA
featuresDF.plot(kind= "bar")                                 




# %% 
# SELECT MODEL 
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=8)




# %% 
# SPLIT DATASET 
import numpy as np  
from sklearn.model_selection import train_test_split

X = features
y = labels 

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.33, random_state=42)

print(X_test[:3])




# && 
# TRAIN MODEL
#neigh = KNeighborsClassifier()(n_neighbors=3)
clf.fit(X_train, y_train)  
accuarcy = clf.score(X_train,y_train) 
print("accuarcy on train data {:.2}%".format(accuarcy))




# %%
# TEST MODEL
accuarcy = clf.score(X_test,y_test) 
print("accuarcy on test data {:.2}%".format(accuarcy))



# %%
# SAVE MODEL 
from joblib import dump, load
filename = "myFirstSavedModel.joblib"
dump(clf, filename) 




## %%
## LOAD MODEL
clfUploaded = load(filename) 




# %%
# TEST WITH UPLOADED MODEL
accuarcy = clfUploaded.score(X_test,y_test) 
print("accuarcy on test data {:.2}%".format(accuarcy))
