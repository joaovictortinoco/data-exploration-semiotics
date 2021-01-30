from deap import graph
from src.models.genetic_programming import GeneticProgramming
from pygraphviz import AGraph

if __name__ == "__main__":

    gp = GeneticProgramming()

    pset = gp.setPrimitive()

    pset = gp.renameArgs(pset)

    tree = gp.generateTree()

    toolbox = gp.createIndividuals((-1,1))

    nodes, edges, labels = graph(expr)

    g = AGraph()

    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog='dot')

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw('tree.pdf')