import pytest
from network_evaluation_tools import data_import_tools as dit
from network_evaluation_tools import network_evaluation_functions as nef
from network_evaluation_tools import network_propagation as prop
import pandas as pd
import numpy as np

def test_example():
    network = dit.load_network_file('../Data/YoungvsOld_UP.csv', delimiter=',', verbose=True)
    genesets = dit.load_node_sets('../Data/DisGeNET_genesets.txt')
    genesets = genesets.items()[0]
    genesets = {genesets[0]:genesets[1]}
    genesets_p = nef.calculate_p(network, genesets)  # calculate the sub-sampling rate p for each node set
    alpha = prop.calculate_alpha(network) # Calculate the Network Alpha
    kernel = nef.construct_prop_kernel(network, alpha=alpha, verbose=True)  # Propagate using the random walk model
    # ## The propagation kernel is a matrix that encodes the pairwise similarities or influences between the nodes in
    # ## the network, based on the connectivity patterns and the propagation parameters. The kernel is typically used to
    # ## propagate information or signals between nodes in the network, by multiplying it with a vector of node-wise
    # ## values or features.
    AUPRC_values = nef.small_network_AUPRC_wrapper(kernel, genesets, genesets_p, n=30, cores=1, verbose=True)
    assert AUPRC_values[0]>0
    one = AUPRC_values[0]
    null_AUPRCs = []
    for i in range(10):
        shuffNet = nef.shuffle_network(network, max_tries_n=10, verbose=True)
        shuffNet_kernel = nef.construct_prop_kernel(shuffNet, alpha=alpha, verbose=False)
        shuffNet_AUPRCs = nef.small_network_AUPRC_wrapper(shuffNet_kernel, genesets, genesets_p, n=30, cores=4,
                                                          verbose=False)
        null_AUPRCs.append(shuffNet_AUPRCs)
        print 'shuffNet', repr(i + 1), 'AUPRCs calculated'

