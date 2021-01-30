import operator

from deap.gp import PrimitiveSetTyped, PrimitiveTree, genFull, graph
from deap import creator, base, tools


class GeneticProgramming():

    def if_then_else(self, input, output1, output2):
        return output1 if input else output2

    def setPrimitive(self):
        pset = PrimitiveSetTyped("main", [bool, float], float)
        pset.addPrimitive(operator.xor, [bool, bool], bool)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(self.if_then_else, [bool, float, float], float)
        pset.addTerminal(3.0, float)
        pset.addTerminal(1, bool)

        return pset

    def generateTree(self, pset, min, max):
        return genFull(pset, min_=min, max_=max)

    def createIndividuals(self, weights, pset):
        creator.create("Fitness", base.Fitness, weights=(-1, 1))
        creator.create("Individuals", PrimitiveTree, fitness=creator.Fitness, pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", genFull, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individuals, toolbox.expr)

        return toolbox.individual()

    def evaluate(self, individual):
        
        return ''
