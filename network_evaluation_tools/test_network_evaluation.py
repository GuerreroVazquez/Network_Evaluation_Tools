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
              'Intracranial Aneurysm': float('inf'),
              'Muscle Weakness': float('inf')}
alpha = 0.684


def test_general_function():
    """
    This test evaluates the conjuction of rhe scores and randomWalk to make sure the results keep
    the same values as intended
    :return:
    """
    _network = dit.load_network_file(network_test_file, delimiter=',', verbose=True)
    _gene_sets = dit.load_node_sets(disease_test_file)
    _gene_sets_p = nef.calculate_p(_network, _gene_sets)  # calculate the sub-sampling rate p for each node set
    _alpha = prop.calculate_alpha(_network)  # Calculate the Network Alpha
    kernel = nef.construct_prop_kernel(_network, alpha=_alpha, verbose=True)  # Propagate using the random walk model
    with open('kernel.pkl', 'wb') as f:
        pickle.dump(kernel, f)
    # ## The propagation kernel is a matrix that encodes the pairwise similarities or influences between the nodes in
    # ## the network, based on the connectivity patterns and the propagation parameters. The kernel is typically used to
    # ## propagate information or signals between nodes in the network, by multiplying it with a vector of node-wise
    # ## values or features.
    AUPRC_values = nef.small_network_AUPRC_wrapper(kernel, _gene_sets, _gene_sets_p, n=30, cores=1, verbose=True)
    assert len(AUPRC_values) == len(_gene_sets), "The total of diseases on the Are under the curve is different than " \
                                                 "the genes evaluated"
    assert str(_alpha).isnumeric()
    assert AUPRC_values['Carcinoma, Lewis Lung'] == 0.5136054421768708, "The auprc of Carcinome is different"
    assert _gene_sets_p['Carcinoma, Lewis Lung'] == 0.5921, "The optimized ratio of Carcimona is different"

def test_construct_prop_kernel_with_networkx():
    _network_h = dit.load_network_file(network_test_file, delimiter=',', verbose=True)
    _network = nx.read_gpickle(networkx_test_file)
    _gene_sets = dit.load_node_sets(disease_test_file)
    _gene_sets_p = nef.calculate_p(_network, _gene_sets)  # calculate the sub-sampling rate p for each node set
    _alpha = prop.calculate_alpha(_network)  # Calculate the Network Alpha
    kernel = nef.construct_prop_kernel(_network, alpha=_alpha, verbose=True)  # Propagate using the random walk model
    assert isinstance(kernel, pd.DataFrame)

def test_evaluate_performance():
    with open('kernel.pkl', 'rb') as f:
        kernel = pickle.load(f)

    null_AUPRCs = []
    for i in range(10):
        shuffNet = nef.shuffle_network(network, max_tries_n=10, verbose=True)
        shuffNet_kernel = nef.construct_prop_kernel(shuffNet, alpha=alpha, verbose=False)
        shuffNet_AUPRCs = nef.small_network_AUPRC_wrapper(shuffNet_kernel, genesets, genesets_p, n=30, cores=4,
                                                          verbose=False)
        null_AUPRCs.append(shuffNet_AUPRCs)

    assert len(null_AUPRCs) == 10
    assert null_AUPRCs[0]['Carcinoma, Lewis Lung'] > 0
    assert null_AUPRCs[0]['Follicular adenoma'] < 0


def test_construct_table():
    null_AUPRCs = [pd.Series({'Carcinoma, Lewis Lung': 0.5047183437650458,
                              'Endometrial adenocarcinoma': 0.5047846889952154,
                              'Fanconi Anemia': 0.5044548007882398,
                              'Follicular adenoma': -1.0,
                              'Intracranial Aneurysm': -1.0,
                              'Muscle Weakness': -1.0}),
                   pd.Series(
                       {'Carcinoma, Lewis Lung': 0.51,
                        'Endometrial adenocarcinoma': 0.5,
                        'Fanconi Anemia': 0.5,
                        'Follicular adenoma': -1.0,
                        'Intracranial Aneurysm': -1.0,
                        'Muscle Weakness': -1.0})
                   ]
    null_AUPRCs_table = pd.concat(null_AUPRCs, axis=1)
    null_AUPRCs_table.columns = ['shuffNet' + repr(i + 1) for i in range(len(null_AUPRCs))]
    print(null_AUPRCs_table)
    network_performance = nef.calculate_network_performance_score(pd.Series(AUPRC_values), null_AUPRCs_table,
                                                                  verbose=True,
                                                                  save_path='../Data/Network_Performance.csv')
    network_performance.name = 'Test Network'
    f_network_performance = network_performance  # .loc[lambda x: x > 0 | x < 0]
    print(f_network_performance)
    network_perf_gain = nef.calculate_network_performance_gain(
        pd.Series(AUPRC_values), null_AUPRCs_table, verbose=True, save_path='../Data/Network_Performance_Gain.csv')
    network_perf_gain.name = 'Test Network'
    with open('performance.pkl', 'wb') as f:
        pickle.dump(network_performance, f)
    with open('gain.pkl', 'wb') as f:
        pickle.dump(network_perf_gain, f)


def test_rank_among_other_networks():
    with open('performance.pkl', 'rb') as f:
        network_performance = pickle.load(f)
    with open('gain.pkl', 'rb') as f:
        network_perf_gain = pickle.load(f)
    # Rank network on average performance across gene sets vs performance on same gene sets in previous network set
    all_network_performance = pd.read_csv('../Data/Network_Performance.csv', index_col=0)

    x = all_network_performance.loc[network_performance.index]

    all_network_performance_filt = pd.concat([network_performance, x], axis=1)
    network_performance_rank_table = all_network_performance_filt.rank(axis=1, ascending=False)
    network_performance_rankings = network_performance_rank_table['Test Network']
    # Rank network on average performance gain across gene sets vs performance gain on same gene sets in previous
    # network set
    all_network_perf_gain = pd.read_csv('../Data/Network_Performance_Gain.csv', index_col=0)
    all_network_perf_gain_filt = pd.concat([network_perf_gain, all_network_perf_gain.ix[network_perf_gain.index]],
                                           axis=1)
    network_perf_gain_rank_table = all_network_performance_filt.rank(axis=1, ascending=False)
    network_perf_gain_rankings = network_perf_gain_rank_table['Test Network']
    # Network Performance
    network_performance_metric_ranks = pd.concat(
        [network_performance, network_performance_rankings, network_perf_gain, network_perf_gain_rankings], axis=1)
    network_performance_metric_ranks.columns = ['Network Performance', 'Network Performance Rank',
                                                'Network Performance Gain', 'Network Performance Gain Rank']
    network_performance_metric_ranks.sort_values(
        by=['Network Performance Rank', 'Network Performance', 'Network Performance Gain Rank',
            'Network Performance Gain'],
        ascending=[True, False, True, False])
    # Construct network summary table
    network_summary = {}
    network_summary['Nodes'] = int(len(network.nodes()))
    network_summary['Edges'] = int(len(network.edges()))
    network_summary['Avg Node Degree'] = np.mean(list(dict(network.degree()).values()))
    network_summary['Edge Density'] = 2 * network_summary['Edges'] / float(
        (network_summary['Nodes'] * (network_summary['Nodes'] - 1)))
    network_summary['Avg Network Performance Rank'] = network_performance_rankings.mean()
    network_summary['Avg Network Performance Rank, Rank'] = int(
        network_performance_rank_table.mean().rank().ix['Test Network'])
    network_summary['Avg Network Performance Gain Rank'] = network_perf_gain_rankings.mean()
    network_summary['Avg Network Performance Gain Rank, Rank'] = int(
        network_perf_gain_rank_table.mean().rank().ix['Test Network'])
    for item in ['Nodes', 'Edges', 'Avg Node Degree', 'Edge Density', 'Avg Network Performance Rank',
                 'Avg Network Performance Rank, Rank',
                 'Avg Network Performance Gain Rank', 'Avg Network Performance Gain Rank, Rank']:
        print(item + ':\t' + repr(network_summary[item]))


def test_construct_prop_kernel():
    """
    This test contructs the kernel based on a specific network \
    of 206 nodes. If the network for example changes, make sure to
    eddit the last 2 assetions on this test.


    :return:
    """
    _network = dit.load_network_file(network_test_file, delimiter=',', verbose=True)
    _gene_sets = dit.load_node_sets(disease_test_file)
    _gene_sets_p = nef.calculate_p(_network, _gene_sets)  # calculate the sub-sampling rate p for each node set
    _alpha = prop.calculate_alpha(_network)  # Calculate the Network Alpha
    kernel = nef.construct_prop_kernel(_network, alpha=_alpha, verbose=True)
    assert isinstance(kernel, pd.DataFrame)
    assert kernel.shape == (len(_network.nodes), len(_network.nodes))  # Propagate using the random walk model
