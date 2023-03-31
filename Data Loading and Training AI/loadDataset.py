from sklearn.datasets import load_iris
dataSet = load_iris()

features = dataSet.data
labels = dataSet.target                                       #dataset in i.inde ki adları verir#
labelsNames = list(dataSet.target_names)
featuresNames = dataSet.feature_names                         # çeşit adı, cinsi ve boyutlatı feature_names ile çağırılır

print([labelsNames[i] for i in labels[47:52]])                # verinin içinde hangi aralıklarda ne olduğu
print(featuresNames)
