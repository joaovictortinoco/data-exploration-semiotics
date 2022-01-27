import statistics

import sklearn.metrics
from sklearn.neural_network import MLPClassifier
from time import process_time


def createInstance(X_train, X_test, y_train, y_test):

    mlp_time_avg = 0
    fscore_avg = []
    accuracy_avg = []
    for i in range(0,29):
        mlp_time_start = process_time()
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=None, max_iter=1000).fit(
            X_train, y_train)

        predict_X_test = classifier.predict(X_test)
        predict_X_train = classifier.predict(X_train)
        fscore_avg.append(sklearn.metrics.f1_score(y_test, predict_X_test))
        accuracy_avg.append(sklearn.metrics.accuracy_score(y_test, predict_X_test))

        mlp_time_end = process_time()
        mlp_time_avg += mlp_time_end-mlp_time_start

    print('MLP f_score: ', sum(fscore_avg)/30)
    print('MLP accuracy: ', sum(accuracy_avg)/30)
    print('MLP std f_score: ', statistics.pstdev(fscore_avg))
    print('MLP std accuracy: ', statistics.pstdev(accuracy_avg))
    print('MLP time processing: ', mlp_time_avg/30)

    return predict_X_test, predict_X_train, classifier, mlp_time_end-mlp_time_start

