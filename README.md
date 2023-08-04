# Iris Classification Project
The purpose of this project is to illustrate the classification model of Iris flowers.








# LOAD DATASET
First of all, we upload our data.

> from sklearn.datasets import load_iris
>dataSet = load_iris()
>
>features = dataSet.data
>labels = dataSet.target                                      
>labelsNames = list(dataSet.target_names)
>featuresNames = dataSet.feature_names                         
>
>print([labelsNames[i] for i in labels[47:52]])               
>print(featuresNames) 




# ANALAYZE DATA 
We use Pandas to analyze the data we upload.

>import pandas as pd
>
>print(type(features))
>
>featuresDF= pd.DataFrame(features)
>featuresDF.columns = featuresNames
>
>print(type(featuresDF))
>print(featuresDF.describe())                                
>print(featuresDF.info())  




# VISUALIZE DATA 
Then we visualized the data we uploaded.
>featuresDF.plot(x="sepal lenght" (cm), y= "sepal width (cm)", kind= "scatter") 

or

>featuresDF.plot(kind= "bar")

You can access more from https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html



# SELECT MODEL 
First we chose our model from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html, then we selected it as KNeighborsClassifier and ran it with parameters.

>from sklearn.neighbors import KNeighborsClassifier
>
>clf = KNeighborsClassifier(n_neighbors=8) 



# SPLIT DATASET 
After choosing our model, we divided our data with ***train_test_split***.

>import numpy as np  
>from sklearn.model_selection import train_test_split
>X, y = np.arange(10).reshape((5, 2)), range(5)
>
>X = features
>y = labels 
>
>X_train, X_test, y_train, y_test = train_test_split(
>   X, y, test_size=0.33, random_state=42)

 Now we have training and testing data and the right labels.
 
 
 
 
 # TRAIN MODEL 
>clf.fit(X_train, y_train)  
>
>accuarcy = clf.score(X_train,y_train)
>
>print("accuarcy on train data {:.2}%".format(accuarcy))

We have seen the success of this model we have trained.



# TEST MODEL
We carry out the test
>accuarcy = clf.score(X_test,y_test) 
>
>print("accuarcy on test data {:.2}%".format(accuarcy))


# SAVE MODEL 
We stored the model using .joblib. we loaded the same function into a new variable with load while storing it with dump.

>from joblib import dump, load
>filename = "myFirstSavedModel.joblib"
>dump(clf, filename)
>
>clfUploaded = load(filename)



# TEST WITH UPLOADED MODEL
We're testing this model again with the same data set 

>accuarcy = clfUploaded.score(X_test,y_test)
>
>print("accuarcy on test data {:.2}%".format(accuarcy))
