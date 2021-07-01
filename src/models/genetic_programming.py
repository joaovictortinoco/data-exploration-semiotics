import itertools
import operator
import random

from deap.gp import PrimitiveSetTyped, PrimitiveTree, genFull, graph, genHalfAndHalf
from deap import creator, base, tools

from src.utils.file_parser import convertToArrayData


def if_then_else(input, output1, output2):
    return output1 if input else output2

def division(topNumber, bottomNumber):
    try:
        return topNumber/bottomNumber
    except ZeroDivisionError: return 1

def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2

class GeneticProgramming:

    def setPrimitive(self):
        pset = PrimitiveSetTyped("main", itertools.repeat(float, 5), str, "IRIS")

        pset.addPrimitive(operator.add, [float, float, float, float], str)
        pset.addPrimitive(operator.sub, [float, float, float, float], str)
        pset.addPrimitive(operator.mul, [float, float, float, float], str)
        # pset.addPrimitive(division, [float, float], str)

        # pset.addPrimitive(operator.lt, [float, float], int)
        # pset.addPrimitive(operator.gt, [float, float], int)
        # pset.addPrimitive(operator.eq, [float, float], int)
        # pset.addPrimitive(if_then_else, [bool, float, float], float)

        pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
        pset.addTerminal(False, bool)

        return pset

    def generateTree(self, pset, min, max):
        return genFull(pset, min_=min, max_=max)

    def createIndividuals(self, pset):
        creator.create("Fitness", base.Fitness, weights=(0,10))
        creator.create("Individuals", PrimitiveTree, fitness=creator.Fitness, pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individuals, toolbox.expr)
        # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", compile, pset=pset)

        return toolbox

    def evaluate(self, toolbox, expr):
        function = expr
        iris_sample = convertToArrayData('files/genetic_programming/database/iris/iris.data')
        result = sum(bool(function(*plants[:5])) is 'Iris-setosa' for plants in iris_sample)

        print(result)
        # return result

