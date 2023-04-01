clf.fit(X_train, y_train)

accuarcy = clf.score(X_train,y_train)

print("accuarcy on train data {:.2}%".format(accuarcy))
