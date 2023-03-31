import pandas as pd

print(type(features))

featuresDF= pd.DataFrame(features)
featuresDF.columns = featuresNames

print(type(featuresDF))
print(featuresDF.describe())                                 
print(featuresDF.info()) 
