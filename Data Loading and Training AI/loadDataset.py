from sklearn.datasets import load_iris
dataSet = load_iris()

features = dataSet.data
labels = dataSet.target                                       
labelsNames = list(dataSet.target_names)
featuresNames = dataSet.feature_names                        

print([labelsNames[i] for i in labels[47:52]])                
print(featuresNames)
