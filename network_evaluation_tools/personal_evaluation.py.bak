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
    averages = dict.fromkeys(gene_sets.keys(), 0)
    for element in null_AUPRCs:
        for key, value in element.items():
            averages[key] = averages[key] + value
    for key, value in averages.items():
        averages[key] = averages[key] / len(null_AUPRCs)

    for key, value in AUPRC_values.items():
        print key, value, [random[key] for random in null_AUPRCs], averages[key]


if __name__ == "__main__":
    main()
