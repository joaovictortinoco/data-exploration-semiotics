from sklearn import tree


def createInstance(X_train, X_test, y_train):
    classifier = tree.DecisionTreeClassifier(random_state=None)
    classifier.fit(X_train, y_train)

    return classifier.predict(X_test), classifier.predict(X_train), classifier
