# Iris-Classification-Project
The purpose of this educational project is to illustrate the classification model of Iris flowers.




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
