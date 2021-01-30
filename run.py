from deap.gp import graph
from src.models.genetic_programming import GeneticProgramming
from pygraphviz import AGraph


def main():
    print('Instanciando GP')
    gp = GeneticProgramming()

    print('Set das primitivas e terminais')
    pset = gp.setPrimitive()

    pset.renameArguments(ARG0='x')
    pset.renameArguments(ARG1='y')

    print('Criando os indivíduos')
    expr = gp.createIndividuals((-1, 1), pset)

    nodes, edges, labels = graph(expr)

    g = AGraph()

    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog='dot')

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw('tree.pdf')


if __name__ == "__main__":
    main()
