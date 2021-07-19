from sklearn.neural_network import MLPClassifier


def createInstance(X_train, X_test, y_train):
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1, max_iter=1000).fit(
        X_train, y_train)

    return classifier.predict(X_test), classifier.predict(X_train)

