import numpy as np
import sklearn.metrics

from src.utils import fetch_dataset
from src.models.classifiers import mlp

import math
import operator
import random

from deap import algorithms
from deap import gp
from deap import creator, base, tools

X_train = []
X_test = []
y_train = []
y_test = []
blackbox_prediction_test = []
blackbox_prediction_train = []
toolbox = None


def protectedDiv(left, right):
    try:
        if right == 0:
            right = 1
        return left / right
    except ZeroDivisionError:
        return 1


def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    global toolbox
    global y_train
    func = toolbox.compile(expr=individual)

    y_pred = []

    # Evaluate the mean squared error between the expression and the real function from regression tree
    for x in enumerate(X_train):
        print(*x)
        function_result = int(func(*x) > 0.5)
        y_pred.append(function_result)

    return sklearn.metrics.f1_score(y_train, y_pred),


def interpretMLP(individual):
    # Transform the tree expression in a callable function
    global toolbox
    global blackbox_prediction_train
    func = toolbox.compile(expr=individual)

    y_pred = []

    for x in enumerate(X_train):
        function_result = int(func(*x[1]) > 0.5)
        y_pred.append(function_result)

    return sklearn.metrics.f1_score(blackbox_prediction_train, y_pred),


print('Generation of GP set...')

pset = gp.PrimitiveSet("MAIN", 64)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", interpretMLP)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def calculateScore(individuals):
    print('Calculating Score from MLP prediction test...')
    global blackbox_prediction_test
    hallOfFame = []

    for i in individuals:
        func = toolbox.compile(expr=i)
        y_gp = []
        for x in enumerate(X_test):
            function_result = int(func(*list(x[1])) > 0.5)
            y_gp.append(function_result)

        gp_f1score = sklearn.metrics.f1_score(blackbox_prediction_test, np.array(y_gp))
        hallOfFame.append((gp_f1score, str(i)))

    hallOfFame.sort(key=lambda x: x[0], reverse=True)
    print('Best Function:', hallOfFame[0])

    mlp_fscore = sklearn.metrics.f1_score(y_test, blackbox_prediction_test)
    print('MLP Score:', mlp_fscore)


def main():
    global X_train, X_test, y_train, y_test, toolbox, blackbox_prediction_test, blackbox_prediction_train
    X_train, X_test, y_train, y_test = fetch_dataset.fetch_digits()

    blackbox_prediction_test, blackbox_prediction_train = mlp.createInstance(X_train, X_test, y_train)

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
