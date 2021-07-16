import math
import operator
import random
import time

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import pandas as pd

# print('Get California Housing Dataset...')
# california_dataset = fetch_california_housing()
# print(california_dataset)
# print(california_dataset.target)

print('Opening wine dataset...')
wine_dataset = pd.read_csv('./files/genetic_programming/database/winequality-red.csv', sep=';')

X = pd.DataFrame(wine_dataset.values, columns=wine_dataset.columns)
y = X.iloc[:,-1]
X = X.iloc[:,:-1]
print(X)
print(y.values)

print('Target test and train data...')
X_train, X_test, y_train, y_test = train_test_split(X, y.values, random_state=1, test_size=0.2)
#
scaler_X =  StandardScaler()
print('Normalize dataset...')
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

print('MLP creation and training...')
regression = MLPRegressor(hidden_layer_sizes=(64,64,64), activation="logistic", random_state=1, max_iter=2000)\
    .fit(X_train_scaled, y_train)
#
print('MLP prediction...')
y_prediction = regression.predict(X_test_scaled)
print(y_prediction)

def protectedDiv(left, right):
    try:
        if right == 0:
            right = 1
        return left / right
    except ZeroDivisionError:
        time.sleep(10)
        return 1

print('Generation of GP set...')

pset = gp.PrimitiveSet("MAIN", 11)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1))

creator.create("FitnessMin", base.Fitness, weights=(1,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function from regression tree
    square_errors = \
        (
            (
                (func(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])) for x in X_train_scaled
            )
        )
    return math.fsum(square_errors)/len(X_train_scaled),


toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(hof)
    return pop, log, hof


if __name__ == "__main__":
    main()
