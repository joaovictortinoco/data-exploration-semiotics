import statistics

import deap.gp
import numpy
import numpy as np
from time import process_time
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
mlp_time = 0
time_start = 0
time_end = 0
mlp_classifier = None


def protectedDiv(left, right):
    try:
        if right == 0:
            right = 1
        return left / right
    except ZeroDivisionError:
        return 1


# Evaluate fitness of an individual within a generation.
def interpretMLP(individual):
    func = toolbox.compile(expr=individual)
    split_points = 0
    countDivision = 0
    countMult = 0
    countPrimitive = 0
    countTerminals = 0
    for i in individual.copy():
        split_points += i.arity if type(i) == deap.gp.Primitive else 0
        countPrimitive += 1 if type(i) == deap.gp.Primitive else 0
        countTerminals += 1 if type(i) == deap.gp.Terminal else 0
        countDivision += 1 if type(i) == deap.gp.Primitive and i.name == 'protectedDiv' else 0
        countMult += 1 if type(i) == deap.gp.Primitive and i.name == 'mul' else 0
    y_pred = []

    for x in enumerate(X_train):
        function_result = int(func(*x[1]) > 0.5)
        y_pred.append(function_result)

    avgTreeLength = individual.__len__()/split_points if split_points != 0 else 0
    ari = (countTerminals * countPrimitive) + split_points
    complextTerminals = countDivision + countMult

    return sklearn.metrics.f1_score(blackbox_prediction_train, y_pred), avgTreeLength, ari, complextTerminals


def generatePrimitive(n_parameters: int, black_box_function):
    global toolbox

    pset = gp.PrimitiveSet("MAIN", n_parameters)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.renameArguments(ARG0='x', ARG1='y', ARG2='z', ARG3='t')

    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", black_box_function)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))

    return toolbox


def calculateScore(individuals, pareto):
    global blackbox_prediction_test
    hallOfFame = []
    hof_height_sum = []
    hof_node_sum = []

    for i in individuals:
        func = toolbox.compile(expr=i)
        y_gp = []
        for x in enumerate(X_test):
            function_result = int(func(*list(x[1])) > 0.5)
            y_gp.append(function_result)

        gp_f1score = sklearn.metrics.f1_score(blackbox_prediction_test, np.array(y_gp))
        gp_accuracy_score = sklearn.metrics.accuracy_score(blackbox_prediction_test, np.array(y_gp))
        hallOfFame.append((gp_f1score, i, gp_accuracy_score))

    for individual in hallOfFame:
        hof_height_sum.append(individual[1].height)
        hof_node_sum.append(individual[1].__len__())

    hallOfFame.sort(key=lambda x: x[0], reverse=True)

    mlp_fscore = sklearn.metrics.f1_score(y_test, blackbox_prediction_test)
    mlp_accuracy = sklearn.metrics.accuracy_score(y_test, blackbox_prediction_test)

    return hallOfFame[0][0], hallOfFame[0][1], mlp_fscore, hallOfFame[0][2], \
           mlp_accuracy, \
           sum(hof_height_sum) / len(hallOfFame), sum(hof_node_sum) / len(hallOfFame), \
           pareto,


def executeDecisionTree():
    dt_classification_test, dt_classification_train, classifier = decision_tree.createInstance(X_train, X_test, y_train)

    return sklearn.metrics.f1_score(blackbox_prediction_test, dt_classification_test), \
           sklearn.metrics.accuracy_score(blackbox_prediction_test, dt_classification_test), \
           classifier


def executeGeneticProgramming():
    global X_train
    pareto = tools.ParetoFront()

    toolbox = generatePrimitive(len(X_train[0]), interpretMLP)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(10)
    fscore_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    avgTree_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
    ari_stats = tools.Statistics(lambda ind: ind.fitness.values[2])
    complexTerminals_stats = tools.Statistics(lambda ind: ind.fitness.values[3])
    mstats = tools.MultiStatistics(fscore_stats=fscore_stats, avgTree_stats=avgTree_stats, ari_stats=ari_stats, complexTerminals_stats=complexTerminals_stats)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, mstats, halloffame=hof, verbose=True)

    pareto.update(pop)

    return calculateScore(hof, pareto),


def generateReport(n_experiments, best_pareto=None):
    gp_fscore_sum = []
    gp_accuracy_sum = []
    gp_height_sum = []
    gp_node_sum = []

    dt_fscore_sum = []
    dt_accuracy_sum = []
    dt_height_sum = []
    dt_node_sum = []

    mlp_fscore_sum = []
    mlp_accuracy_sum = []

    best_gp_fscore = 0
    best_gp_function = None
    gp_sum_time = 0
    dt_sum_time = 0
    best_pareto
    accumulatedPareto = []
    for i in range(0, n_experiments - 1):
        time_start = process_time()
        gp_fscore, gp_function, mlp_fscore, accuracy_score, mlp_accuracy, gp_height, gp_node, pareto_ = \
            executeGeneticProgramming()[0]
        time_end = process_time()
        gp_sum_time += time_end - time_start

        accumulatedPareto.append(pareto_)

        time_dt_start = process_time()
        dt_fscore, dt_accuracy, classifier = executeDecisionTree()
        time_dt_end = process_time()
        dt_sum_time = time_dt_end - time_dt_start
        dt_height_sum.append(classifier.get_depth())
        dt_node_sum.append(classifier.get_n_leaves())

        mlp_fscore_sum.append(mlp_fscore)
        mlp_accuracy_sum.append(mlp_accuracy)

        gp_fscore_sum.append(gp_fscore)
        gp_accuracy_sum.append(accuracy_score)
        gp_height_sum.append(gp_height)
        gp_node_sum.append(gp_node)

        dt_fscore_sum.append(dt_fscore)
        dt_accuracy_sum.append(dt_accuracy)

        if gp_fscore > best_gp_fscore:
            best_gp_fscore = gp_fscore
            best_gp_function = gp_function
            best_pareto = pareto_

    # generateTree(best_gp_function, best_pareto)
    generateParetoCharts(accumulatedPareto)
    print('Genetic Programming mean f1_score MLP: ', sum(gp_fscore_sum) / n_experiments)
    print('Genetic Programming mean accuracy_score MLP:', sum(gp_accuracy_sum) / n_experiments)
    print('Genetic Programming F1_score Std:', statistics.pstdev(gp_fscore_sum))
    print('Genetic Programming accuracy Std: ', statistics.pstdev(gp_accuracy_sum))
    print('Genetic Programming Processing time: ', gp_sum_time / n_experiments)
    print('Genetic Programming mean model size: (', sum(gp_height_sum) / n_experiments, ', ',
          sum(gp_node_sum) / n_experiments, ')')
    print('Genetic Programming std model size: (', statistics.pstdev(gp_height_sum), ', ',
          statistics.pstdev(gp_node_sum), ')')

    print('Decision Tree mean f1_score MLP: ', sum(dt_fscore_sum) / n_experiments)
    print('Decision Tree mean accuracy MLP: ', sum(dt_accuracy_sum) / n_experiments)
    print('Decision Tree F1_score Std: ', statistics.pstdev(dt_fscore_sum))
    print('Decision Tree accuracy Std: ', statistics.pstdev(dt_accuracy_sum))
    print('Decision Tree Processing time: ', dt_sum_time / n_experiments)
    print('Decision Tree mean model size: (', sum(dt_height_sum) / n_experiments, ', ',
          sum(dt_node_sum) / n_experiments, ')')
    print('Decision Tree std model size: (', statistics.pstdev(dt_height_sum), ', ', statistics.pstdev(dt_node_sum),
          ')')

    print('MLP mean f1_score y: ', sum(mlp_fscore_sum) / n_experiments)
    print('MLP std f1_score y: ', statistics.pstdev(mlp_fscore_sum) / n_experiments)
    print('MLP mean accuracy y: ', sum(mlp_accuracy_sum) / n_experiments)
    print('MLP std accuracy y: ', statistics.pstdev(mlp_accuracy_sum) / n_experiments)
    # print('Processing time MLP: ', mlp_time)


def generateParetoCharts(paretos):
    import matplotlib.pyplot as plt

    fscoreData = []
    avgTreeData = []
    ariData = []
    complexTermData = []
    for p in paretos:
        for i in p.items:
            fscore, avgTreeLength, ari, complexTerminals = interpretMLP(i)
            fscoreData.append(fscore)
            avgTreeData.append(avgTreeLength)
            ariData.append(ari)
            complexTermData.append(complexTerminals)

    ax = plt.axes(projection='3d')
    zline = complexTermData
    xline = ariData
    yline = fscoreData
    ax.scatter(xline, yline, zline, c=zline, label=['ari', 'fscore', 'complexTerminals'], cmap='viridis', lineWidth=0.5)
    ax.set_xlabel('ari', fontweight='bold')
    ax.set_ylabel('fscore', fontweight='bold')
    ax.set_zlabel('complex terminals', fontweight='bold')

    fig, ax = plt.subplots()
    ax.plot(avgTreeData, fscoreData, 'ro')

    ax.set(xlabel='avg tree length', ylabel='fscore',
           title='')
    ax.grid()

    fig.savefig("pareto_results/avgTreeLength.png")
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(ariData, fscoreData, 'ro')

    ax1.set(xlabel='automated readability index', ylabel='fscore',
           title='')
    ax1.grid()

    fig1.savefig("pareto_results/ari.png")
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(complexTermData, fscoreData, 'ro')

    ax2.set(xlabel='complex terminals', ylabel='fscore',
            title='')
    ax2.grid()

    fig2.savefig("pareto_results/complexTerminals.png")
    plt.show()


def generateTree(best_gp_function, best_pareto):
    import pygraphviz as pgv

    # nodes, edges, labels = gp.graph(best_gp_function)
    #
    # g = pgv.AGraph()
    # g.add_nodes_from(nodes)
    # g.add_edges_from(edges)
    # g.layout(prog="dot")
    #
    # for i in nodes:
    #     n = g.get_node(i)
    #     n.attr["label"] = labels[i]
    #
    # g.draw("hof_results/tree_hall_of_fame.pdf")

    pareto_draw = sorted(best_pareto.items, key=lambda i: i.__len__())

    for i_pareto in pareto_draw:
        nodes, edges, labels = gp.graph(i_pareto)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw("pareto_results/tree_pareto_" + str(pareto_draw.index(i_pareto)) + ".pdf")

def main():
    # Get global scope variables
    global X_train, X_test, y_train, y_test, toolbox, blackbox_prediction_test, blackbox_prediction_train, mlp_time

    # Fetch dataset and set train/test variables
    X_train, X_test, y_train, y_test = fetch_dataset.fetch_ionosphere()

    # Execute blackbox algorithm
    blackbox_prediction_test, blackbox_prediction_train, classifier, mlp_time = mlp.createInstance(X_train, X_test,
                                                                                                   y_train, y_test)

    generateReport(n_experiments=30)


if __name__ == '__main__':
    main()
