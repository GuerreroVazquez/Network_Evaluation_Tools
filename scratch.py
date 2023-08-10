import itertools

import pandas as pd
from numpy import nan
import matplotlib_venn
from matplotlib_venn import venn2
from matplotlib import pyplot as plt

# %matplotlib inline

# Read the CSV file
data = pd.read_csv("/home/karen/Documents/GitHub/Differential_expression/Results/up_regulated_all.csv")

# Extract group names from the header
group_columns = [column.split('_')[-1] for column in data.columns]

# Initialize empty dictionaries to store the intersection and union results
intersection_results = {}
union_results = {}


def get_values(experiments):
    experiment_sets = {}
    for experiment in experiments:
        experiment_sets[experiment] = set()
        for column in data.columns:
            if experiment in column:
                experiment_sets[experiment].update(set(data[column]))
    return experiment_sets


def get_experiments_groups():
    experiments = ["GSE152558", "GSE157585", "GSE164471", "GSE875105", "GSE23527"]
    experiment_sets = get_values(experiments)
    return experiment_sets


def get_age_groups():
    age_groups = ["YM", "MO", "YO"]
    age_sets = get_values(age_groups)
    return age_sets


def get_age_groups_and_slope():
    age_groups = ["YM_UP", "MO_UP", "YO_UP", "YM_Down", "MO_Down", "YO_Down"]
    age_sets = get_values(age_groups)
    return age_sets


def get_intersections(sets_dictionary):
    intersections = {}
    for r in range(2, len(sets_dictionary) + 1):
        combinations = list(itertools.combinations(sets_dictionary.keys(), r))
        for combo in combinations:
            sets = []
            for c in combo:
                sets.append(sets_dictionary[c])
            intersection = set.intersection(*sets)
            intersections[combo] = intersection
    return intersections


def intersection_of_experiment():
    sets_dictionary = get_experiments_groups()
    intersections = get_intersections(sets_dictionary=sets_dictionary)
    non_empty = filter_non_empty(intersections=intersections)
    return non_empty


def intersection_of_age_group():
    sets_dictionary = get_age_groups()
    intersections = get_intersections(sets_dictionary=sets_dictionary)
    non_empty = filter_non_empty(intersections=intersections)
    return non_empty


def intersection_of_age_group_n_slopes():
    sets_dictionary = get_age_groups_and_slope()
    intersections = get_intersections(sets_dictionary=sets_dictionary)
    non_empty = filter_non_empty(intersections=intersections)
    return non_empty


def filter_non_empty(intersections):
    non_empty = {}
    for key, value in intersections.items():
        value.remove(nan)
        if len(value) > 0:
            non_empty[key] = value
    return non_empty


def draw_vennDiagram_2_experiments():
    groups_sets = {}
    non_empty_ex = intersection_of_experiment()
    experiments = get_experiments_groups()
    print_venns(intereste_venns=non_empty_ex, sets=experiments)
def draw_vennDiagram_2_ages():
    groups_sets = {}
    non_empty_ex = intersection_of_age_group()
    experiments = get_age_groups()
    print_venns(intereste_venns=non_empty_ex, sets=experiments)

def draw_vennDiagram_2_ages_n_slopes():
    groups_sets = {}
    non_empty_ex = intersection_of_age_group_n_slopes()
    experiments = get_age_groups_and_slope()
    print_venns(intereste_venns=non_empty_ex, sets=experiments)

def print_venns(intereste_venns, sets):
    for groups, intersection in intereste_venns.items():
        a = groups[0]
        b = groups[1]
        C = len(intersection)
        A = len(sets[a]) - C
        B = len(sets[b]) - C
        venn2(subsets=(A, B, C), set_labels=(a,b))

def print_all_venns():
    draw_vennDiagram_2_experiments()
    draw_vennDiagram_2_ages()
    draw_vennDiagram_2_ages_n_slopes()

# Compute intersections of every pair of EXPERIMENT groups
experiment_groups = list(set([group for group in group_columns if 'EXPERIMENT' in group]))
for i in range(len(experiment_groups) - 1):
    for j in range(i + 1, len(experiment_groups)):
        group1 = data[f"EXPERIMENT_{experiment_groups[i]}"]
        group2 = data[f"EXPERIMENT_{experiment_groups[j]}"]

        intersection = set(group1) & set(group2)

        intersection_results[f"EXPERIMENT_{experiment_groups[i]}_{experiment_groups[j]}"] = intersection

# Compute intersections of every pair of TIMEFRAME groups
timeframe_groups = list(set([group for group in group_columns if 'TIMEFRAME' in group]))
for i in range(len(timeframe_groups) - 1):
    for j in range(i + 1, len(timeframe_groups)):
        group1 = data[f"TIMEFRAME_{timeframe_groups[i]}"]
        group2 = data[f"TIMEFRAME_{timeframe_groups[j]}"]

        intersection = set(group1) & set(group2)

        intersection_results[f"TIMEFRAME_{timeframe_groups[i]}_{timeframe_groups[j]}"] = intersection

# Compute intersections of TIMEFRAME and SLOPE groups
timeframe_slope_groups = list(set([group for group in group_columns if 'TIMEFRAME_SLOPE' in group]))
for group in timeframe_slope_groups:
    intersection_results[f"TIMEFRAME_SLOPE_{group}"] = set(data[f"TIMEFRAME_SLOPE_{group}"])

# Print the intersection results
print("Intersection Results:")
for key, value in intersection_results.items():
    print(key, ":", value)
