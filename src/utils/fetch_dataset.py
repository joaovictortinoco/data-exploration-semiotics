from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def fetch_digits():
    print('Opening digits dataset...')

    digits_dataset = load_digits(n_class=2)

    X = pd.DataFrame(digits_dataset.data, columns=digits_dataset.feature_names)

    y = digits_dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    fetch_digits()
