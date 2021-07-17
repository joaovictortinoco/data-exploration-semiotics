from sklearn.neural_network import MLPClassifier


def createInstance(X_train, X_test, y_train):
    print('MLP creation and training...')
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1, max_iter=1000).fit(
        X_train, y_train)

    return classifier.predict(X_test)

