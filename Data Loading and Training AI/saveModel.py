from joblib import dump, load
filename = "myFirstSavedModel.joblib"
dump(clf, filename) 

clfUploaded = load(filename) 
