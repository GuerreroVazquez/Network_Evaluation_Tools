import pytest
from network_evaluation_tools import data_import_tools as dit
from network_evaluation_tools import network_evaluation_functions as nef
from network_evaluation_tools import network_propagation as prop
import pandas as pd
import numpy as np

def test_example():
    network = dit.load_network_file('../Data/YoungvsOld_UP.csv', delimiter=',', verbose=True)
    genesets = dit.load_node_sets('../Data/DisGeNET_genesets.txt')
    genesets_p = nef.calculate_p(network, genesets)
    alpha = prop.calculate_alpha(network)
    kernel = nef.construct_prop_kernel(network, alpha=alpha, verbose=True)
    AUPRC_values = nef.small_network_AUPRC_wrapper(kernel, genesets, genesets_p, n=30, cores=1, verbose=True)
