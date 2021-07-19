import numpy as np
import sklearn.metrics

from src.utils import fetch_dataset
from src.models.classifiers import mlp, decision_tree

import operator

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

    for x in enumerate(X_train):
        function_result = int(func(*x) > 0.5)
        y_pred.append(function_result)

    return sklearn.metrics.f1_score(y_train, y_pred),


def interpretMLP(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    y_pred = []

    for x in enumerate(X_train):
        function_result = int(func(*x[1]) > 0.5)
        y_pred.append(function_result)

    return sklearn.metrics.f1_score(blackbox_prediction_train, y_pred),


def generatePrimitive(n_parameters: int, black_box_function):
    global toolbox

    pset = gp.PrimitiveSet("MAIN", n_parameters)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    # pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1))

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", black_box_function)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox


def calculateScore(individuals):
    # print('Calculating Score from MLP and ISGP')
    global blackbox_prediction_test
    hallOfFame = []

    for i in individuals:
        func = toolbox.compile(expr=i)
        y_gp = []
        for x in enumerate(X_test):
            function_result = int(func(*list(x[1])) > 0.5)
            y_gp.append(function_result)

        gp_f1score = sklearn.metrics.f1_score(blackbox_prediction_test, np.array(y_gp))
        hallOfFame.append((gp_f1score, i))

    hallOfFame.sort(key=lambda x: x[0], reverse=True)

    mlp_fscore = sklearn.metrics.f1_score(y_test, blackbox_prediction_test)

    return hallOfFame[0][0], hallOfFame[0][1], mlp_fscore


def executeDecisionTree():
    dt_classification_test, dt_classification_train = decision_tree.createInstance(X_train, X_test, y_train)

    return sklearn.metrics.f1_score(blackbox_prediction_test, dt_classification_test)


def executeGeneticProgramming():
    global X_train
    toolbox = generatePrimitive(len(X_train[0]), interpretMLP)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(10)

    # stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # stats_size = tools.Statistics(len)
    # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # mstats.register("avg", np.mean)
    # mstats.register("max", np.max)
    # mstats.register("std", np.std)
    # mstats.register("min", np.min)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, halloffame=hof, verbose=True)

    return calculateScore(hof)

def generateReport(n_experiments):
    gp_fscore_sum = 0
    mlp_fscore_sum = 0
    dt_fscore_sum = 0
    best_gp_fscore = 0
    best_gp_function = None

    for i in range(0, n_experiments-1):
        gp_fscore, gp_function, mlp_fscore = executeGeneticProgramming()
        dt_fscore = executeDecisionTree()

        if gp_fscore > best_gp_fscore:
            best_gp_fscore = gp_fscore
            best_gp_function = gp_function

        gp_fscore_sum += gp_fscore
        mlp_fscore_sum += mlp_fscore
        dt_fscore_sum += dt_fscore

    print('Genetic Programming mean f1_score MLP: ', gp_fscore_sum / n_experiments)
    print('Decision Tree mean f1_score MLP: ', dt_fscore_sum / n_experiments)
    print('MLP mean f1_score y: ', mlp_fscore_sum / n_experiments)

    nodes, edges, labels = gp.graph(best_gp_function)

    ### Graphviz Section ###
    import pygraphviz as pgv

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")



def main():
    # Get global scope variables
    global X_train, X_test, y_train, y_test, toolbox, blackbox_prediction_test, blackbox_prediction_train

    # Fetch dataset and set train/test variables
    X_train, X_test, y_train, y_test = fetch_dataset.fetch_breast_cancer()

    # Execute blackbox algorithm
    blackbox_prediction_test, blackbox_prediction_train = mlp.createInstance(X_train, X_test, y_train)

    # Runs GP along with
    generateReport(n_experiments = 30)


if __name__ == '__main__':
    main()
