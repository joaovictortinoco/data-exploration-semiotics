from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def fetch_iris():
    print('Opening iris dataset...')
    iris_dataset = load_iris()

    print(iris_dataset)

    X = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
    y = iris_dataset.target

    print('Target test and train data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

    scaler_X = StandardScaler()
    print('Normalize dataset...')
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test