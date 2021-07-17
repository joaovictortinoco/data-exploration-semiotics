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
    func = toolbox.compile(expr=individual)
    sum = 0
    diff = 0

    # Evaluate the mean squared error between the expression and the real function from regression tree
    for (index, x) in enumerate(X_train):
        function_result = math.ceil(func(x[0], x[1], x[2], x[3]))
        if function_result < 0: function_result = 0

        diff = (function_result - y_train[index]) ** 2
        sum += diff

    square_error = sum / len(X_train)

    return square_error,


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

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def calculateScore(individuals, blackbox_prediction):
    hallOfFame = []
    for i in individuals:
        func = toolbox.compile(expr=i)
        y_gp = []
        for x in X_test:
            function_result = math.ceil(func(x[0], x[1], x[2], x[3]))
            if function_result < 0: function_result = 0
            y_gp.append(function_result)

        precision_score = sklearn.metrics.precision_score(y_test, np.array(y_gp), average='micro')
        hallOfFame.append((precision_score, i))

    hallOfFame.sort(key=lambda x: x[0])
    print(hallOfFame)

    precision_score_mlp = sklearn.metrics.precision_score(y_test, blackbox_prediction, average='micro')
    print(precision_score_mlp)


def main():
    global X_train, X_test, y_train, y_test, toolbox
    X_train, X_test, y_train, y_test = fetch_dataset.fetch_iris()

    blackbox_prediction = mlp.createInstance(X_train, X_test, y_train)

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

    calculateScore(hof, blackbox_prediction)

    return pop, log, hof


if __name__ == "__main__":
    main()
