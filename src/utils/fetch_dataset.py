from sklearn.datasets import load_iris, load_breast_cancer, load_digits, fetch_olivetti_faces, fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def fetch_iris():
    print('Opening iris dataset...')
    iris_dataset = load_iris()

    X = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

    y = iris_dataset.target

    lcompr = [i for i in range(len(y)) if y[i] > 0]
    X = X.iloc[lcompr]

    y = [int(y[i] > 1) for i in range(len(y)) if y[i] > 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    print(X_train)
    return X_train_scaled, X_test_scaled, y_train, y_test


def fetch_breast_cancer():
    print('Opening breast cancer dataset...')

    breast_dataset = load_breast_cancer()

    X = pd.DataFrame(breast_dataset.data, columns=breast_dataset.feature_names)

    y = breast_dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def fetch_digits(targetNumber1, targetNumber2):

    digits_dataset = load_digits()

    index = (digits_dataset.target == targetNumber1) | (digits_dataset.target == targetNumber2)

    y = digits_dataset.target[index]
    X = pd.DataFrame(digits_dataset.data[index], columns=digits_dataset.feature_names)

    y = np.where(y == targetNumber1, 0, y)
    y = np.where(y == targetNumber2, 1, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def fetch_view_recommendations():
    view_data = pd.read_csv('./src/utils/views_classification.csv')
    X = view_data[
        ['Col_Dimension', 'Col_Measure', 'Col_Function', 'Rows', 'Min', 'Max', 'Distinct', 'Null', 'Deviation']]
    print(X.columns)
    y = view_data[['Class']]
    print(y.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def fetch_kdd():
    # @TODO: Investigar tipos de dados vindo desta base.
    kdd_dataset = fetch_kddcup99();

    X = pd.DataFrame(kdd_dataset.data, columns=kdd_dataset.feature_names)

    print(X.info())

    y = kdd_dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def fetch_ionosphere():
    load_ionosphere = pd.read_csv('./src/utils/ionosphere.data.csv')
    X = load_ionosphere.iloc[:,0:34]
    y = load_ionosphere.iloc[:,34]
    y = np.where(y == 'b', 0, y)
    y = np.where(y == 'g', 1, y)
    y = y.astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def fetch_wine():
    load_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    X = load_wine.iloc[:, 0:12]
    y = load_wine['quality']
    y = np.where(y <= 5, 0, y)
    y = np.where(y > 5, 1, y)
    y = y.astype('int')
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def fetch_banknotes():
    load_banknotes = pd.read_csv('./src/utils/data_banknote_authentication.txt', sep=',')
    X = load_banknotes.iloc[:,0:4]
    y = load_banknotes.iloc[:,4]

    y = y.astype('int')
    X = X.astype('double')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    fetch_banknotes()
