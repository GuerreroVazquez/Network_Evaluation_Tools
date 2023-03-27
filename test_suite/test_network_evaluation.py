import math
import networkx as nx

import pytest
from network_evaluation_tools import data_import_tools as dit
from network_evaluation_tools import network_evaluation_functions as nef
from network_evaluation_tools import network_propagation as prop
import pandas as pd
import numpy as np
import pickle

network_test_file = '../Data/Networks/YoungvsOld_UP.csv'
disease_test_file = '../Data/Evaluations/DisGeNET_genesets.txt'
networkx_test_file = '../Data/NetworkCYJS/graph1_Young_Old_Fuzzy_95.pkl'

AUPRC_values = {'Carcinoma, Lewis Lung': 0.5136054421768708, 'Fanconi Anemia': 0.5048184241212726,
                'Endometrial adenocarcinoma': 0.5036461554318696, 'Follicular adenoma': -1.0,
                'Intracranial Aneurysm': -1.0}
network = dit.load_network_file('../Data/Networks/YoungvsOld_UP.csv', delimiter=',', verbose=True)
genesets = dit.load_node_sets('../Data/Evaluations/DisGeNET_genesets.txt')
genesets = {'Carcinoma, Lewis Lung': genesets['Carcinoma, Lewis Lung'],
            'Fanconi Anemia': genesets['Fanconi Anemia'],
            'Endometrial adenocarcinoma': genesets['Endometrial adenocarcinoma'],
            'Follicular adenoma': genesets['Follicular adenoma'],
            'Intracranial Aneurysm': genesets['Intracranial Aneurysm'],
            'Muscle Weakness': genesets['Muscle Weakness']
            }
genesets_p = {'Carcinoma, Lewis Lung': 0.5921,
              'Fanconi Anemia': 0.5589,
              'Endometrial adenocarcinoma': 0.5921,
              'Follicular adenoma': 0.649,
              'Intracranial Aneurysm': np.NAN,
              'Muscle Weakness': np.NAN}
alpha = 0.684


def test_functionality_small_network_AUPRC_wrapper():
    """
    This test evaluates the conjuction of rhe scores and randomWalk to make sure the results keep
    the same values as intended. This is a functionality test of the function.
    :return:
    """
    dummy_data = {'Source': [0.580046, 0.419954, 0.000000, 0.000000, 0.000000, 0.000000],
                  'Target': [0.419954, 0.580046, 0.000000, 0.000000, 0.000000, 0.000000],
                  'TGFB1': [0.000000, 0.000000, 0.518775, 0.083152, 0.219825, 0.178249],
                  'INHBA': [0.000000, 0.000000, 0.124728, 0.528959, 0.254735, 0.091577],
                  'CXCL8': [0.000000, 0.000000, 0.219825, 0.169824, 0.448953, 0.161398],
                  'Gene4': [0.000000, 0.000000, 0.267373, 0.091577, 0.242098, 0.398952]}
    kernel = pd.DataFrame(dummy_data, columns=['Source', 'Target', 'TGFB1', 'INHBA', 'CXCL8', 'Gene4'],
                          index=['Source', 'Target', 'TGFB1', 'INHBA', 'CXCL8', 'Gene4'])

    # Create the kernel variable
    # kernel = np.array(df)
    # ## The propagation kernel is a matrix that encodes the pairwise similarities or influences between the nodes in
    # ## the network, based on the connectivity patterns and the propagation parameters. The kernel is typically used to
    # ## propagate information or signals between nodes in the network, by multiplying it with a vector of node-wise
    # ## values or features.
    AUPRC_values = nef.small_network_AUPRC_wrapper(kernel, genesets, genesets_p, n=30, cores=1, verbose=True)
    assert len(AUPRC_values) == len(genesets), "The total of diseases on the Are under the curve is different than " \
                                               "the genes evaluated"
    assert np.isclose(AUPRC_values['Carcinoma, Lewis Lung'], 0.72, rtol=0.1), "The AUPRC of Carcinome value is not " \
                                                                              "approximately 0.72"


@pytest.mark.skip(reason="This is a heavy test that does the same as the previous one")
def test_general_function():
    """
    This test evaluates the conjunction of rhe scores and randomWalk to make sure the results keep
    the same values as intended. This is using actual data form real networks
    :return:
    """
    _network = dit.load_network_file(network_test_file, delimiter=',', verbose=True)
    _gene_sets = dit.load_node_sets(disease_test_file)
    _gene_sets_p = nef.calculate_p(_network, _gene_sets)  # calculate the sub-sampling rate p for each node set
    _alpha = prop.calculate_alpha(_network)  # Calculate the Network Alpha
    kernel = nef.construct_prop_kernel(_network, alpha=_alpha, verbose=True)  # Propagate using the random walk model
    # ## The propagation kernel is a matrix that encodes the pairwise similarities or influences between the nodes in
    # ## the network, based on the connectivity patterns and the propagation parameters. The kernel is typically used to
    # ## propagate information or signals between nodes in the network, by multiplying it with a vector of node-wise
    # ## values or features.
    AUPRC_values = nef.small_network_AUPRC_wrapper(kernel, _gene_sets, _gene_sets_p, n=30, cores=1, verbose=True)
    assert len(AUPRC_values) == len(_gene_sets), "The total of diseases on the Are under the curve is different than " \
                                                 "the genes evaluated"
    assert np.isclose(AUPRC_values['Carcinoma, Lewis Lung'], 0.51, rtol=0.1), "The auprc of Carcinome is different"

def test_calculate_p():
    """
    This test checks that the genesets_p will have the same amount of elements than the genesets
    :return:
    """
    _network = nx.read_gpickle(networkx_test_file)
    genesets_p = nef.calculate_p(_network, genesets)
    assert isinstance(genesets_p, dict)
    assert len(genesets_p) == len(genesets)
    pass


def test_calculate_p_no_intersections():
    """
    This test checks that the genesets_p will have the same amount of elements than the genesets.
    The value expected when there is no intersection is NAN, since it is the validation in the rest of
    functions.
    :return:
    """
    _network = nx.read_gpickle(networkx_test_file)
    genesets['Carcinoma, Lewis Lung'] = {'a', 'aa'}
    genesets_p = nef.calculate_p(_network, genesets)
    assert isinstance(genesets_p, dict)
    assert len(genesets_p) == len(genesets)
    assert np.isnan(genesets_p['Carcinoma, Lewis Lung'])

