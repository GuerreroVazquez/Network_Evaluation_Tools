import networkx as nx
import numpy as np
import pandas as pd

from network_evaluation_tools import data_import_tools as dit
from network_evaluation_tools import network_evaluation_functions as nef
from network_evaluation_tools import network_propagation as prop


def main(network_file='../Data/YoungvsOld_UP.csv', disease_file='../Data/DisGeNET_genesets.txt'):
    """
    This function will take a network_file that is the csv with the edges of the network, It can be obtained
    by exporting to file the edge table in cytoscape.
    The disease file is a tsv with the first column being the name of the disease and from there all the genes
    involved in that disease.

    :param network_file: str Name of the file with the network edges. First column should be the nodes sources.
    :param disease_file: str Name of the disease file, each line should be a disease and their genes.
    :return: A dictionary with all the diseases and their AUPRC values on that network.
    """
    network = dit.load_network_file(network_file, delimiter=',', verbose=True)
    gene_sets = dit.load_node_sets(disease_file)
    gene_sets_p = nef.calculate_p(network, gene_sets)  # calculate the sub-sampling rate p for each node set
    alpha = prop.calculate_alpha(network)  # Calculate the Network Alpha
    kernel = nef.construct_prop_kernel(network, alpha=alpha, verbose=True)  # Propagate using the random walk model
    AUPRC_values = nef.small_network_AUPRC_wrapper(kernel, gene_sets, gene_sets_p, n=30, cores=1, verbose=True)
    null_AUPRCs = []
    for i in range(10):
        shuffNet = nef.shuffle_network(network, max_tries_n=10, verbose=True)
        shuffNet_kernel = nef.construct_prop_kernel(shuffNet, alpha=alpha, verbose=False)
        shuffNet_AUPRCs = nef.small_network_AUPRC_wrapper(shuffNet_kernel, gene_sets, gene_sets_p, n=30, cores=4,
                                                          verbose=False)
        null_AUPRCs.append(shuffNet_AUPRCs)
    averages = dict.fromkeys(list(gene_sets.keys()), 0)
    for element in null_AUPRCs:
        for key, value in list(element.items()):
            averages[key] = averages[key] + value
    for key, value in list(averages.items()):
        averages[key] = averages[key] / len(null_AUPRCs)

    for key, value in list(AUPRC_values.items()):
        print(key, value, [random[key] for random in null_AUPRCs], averages[key])


def shuffle_network(network, alpha, genesets, genesets_p, n=30, cores=4):
    null_AUPRCs = []
    for i in range(10):
        shuffNet = nef.shuffle_network(network, max_tries_n=10, verbose=True)
        shuffNet_kernel = nef.construct_prop_kernel(shuffNet, alpha=alpha, verbose=False)
        shuffNet_AUPRCs = nef.small_network_AUPRC_wrapper(shuffNet_kernel, genesets, genesets_p, n=n, cores=cores,
                                                          verbose=False)
        null_AUPRCs.append(shuffNet_AUPRCs)
    null_AUPRCs_table = pd.concat(null_AUPRCs, axis=1)
    null_AUPRCs_table.columns = ['shuffNet' + repr(i + 1) for i in range(len(null_AUPRCs))]
    return null_AUPRCs_table

def get_network_performance(network_name, AUPRC_values, null_AUPRCs_table):
    network_performance = nef.calculate_network_performance_score(AUPRC_values, null_AUPRCs_table, verbose=True)
    network_performance.name = network_name
    network_perf_gain = nef.calculate_network_performance_gain(AUPRC_values, null_AUPRCs_table, verbose=True)
    network_perf_gain.name = network_name
    return network_performance, network_performance

def evaluate_network(network_name, evaluating_network, genesets):
    network = nx.read_gpickle(evaluating_network)
    genesets_p = nef.calculate_p(network, genesets)
    alpha = prop.calculate_alpha(network)
    kernel = nef.construct_prop_kernel(network, alpha=alpha, verbose=True)
    AUPRC_values = nef.small_network_AUPRC_wrapper(kernel, genesets, genesets_p, n=30, cores=1, verbose=True)
    null_AUPRCs_table = shuffle_network(network=network, alpha=alpha, genesets=genesets, genesets_p=genesets_p)
    return get_network_performance(network_name=network_name, AUPRC_values=AUPRC_values,
                                   null_AUPRCs_table=null_AUPRCs_table)


if __name__ == "__main__":
    main()
