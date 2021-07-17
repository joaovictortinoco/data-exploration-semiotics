import math
import operator
import random
import time

import numpy as np
import sklearn.metrics

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

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

print('MLP creation and training...')
regression = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="logistic", random_state=1, max_iter=2000) \
    .fit(X_train_scaled, y_train)

print('MLP prediction...')
y_prediction = regression.predict(X_test_scaled)
print(X_test_scaled)
print(y_prediction)

def protectedDiv(left, right):
    try:
        if right == 0:
            right = 1
        return left / right
    except ZeroDivisionError:
        return 1

print('Generation of GP set...')

pset = gp.PrimitiveSet("MAIN", 4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1))
pset.renameArguments(ARG0='x', ARG1='y', ARG2='z', ARG3='t')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    sum  = 0
    diff = 0
    # Evaluate the mean squared error between the expression and the real function from regression tree
    for (index, x) in enumerate(X_train_scaled):
        diff = (func(x[0], x[1], x[2], x[3]) - y_train[index]) ** 2
        sum += diff

    square_error = sum/len(X_train_scaled)

    return square_error,

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def calculateScore(individuals):
    hallOfFame = []
    for i in individuals:
        func = toolbox.compile(expr=i)
        y_gp = []
        for x in X_test_scaled:
            result = func(x[0], x[1], x[2], x[3])
            y_gp.append(result)

        r2_score = sklearn.metrics.r2_score(y_test, np.array(y_gp))
        hallOfFame.append((r2_score, i))

    hallOfFame.sort(key=lambda x:x[0])
    print(hallOfFame)

    r2_score_mlp = sklearn.metrics.r2_score(y_test, y_prediction)
    print(r2_score_mlp)

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(10)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

    calculateScore(hof)

    return pop, log, hof


if __name__ == "__main__":
    main()
