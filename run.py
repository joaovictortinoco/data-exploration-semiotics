import math
import operator
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score

# Generates decision tree via iris dataset
random_dataset = np.random.RandomState(1)
X = np.sort(5 * random_dataset.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - random_dataset.rand(16))

regression = DecisionTreeRegressor(max_depth=5).fit(X, y)

X_test = np.arange(0.0, 5.0, 1)[:, np.newaxis]

y = regression.predict(X_test)

print('Model Prediction: ', y)


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(0, 5))
pset.renameArguments(ARG0='x')

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
    print(func)
    function_outcome = func(X_test)
    function_outcome[function_outcome >= 1E308] = 0
    # fscore = f1_score(individual_prediction, y)
    # print(fscore)
    # Evaluate the mean squared error between the expression
    # and the real function from regression tree
    # square_errors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in X_test)
    return function_outcome.sum(),


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
    return pop, log, hof


if __name__ == "__main__":
    main()
