"""
    fish.py
    ~~~~~~~

    This module solves the fish homework problem.

    Run with "python fish.py"

    Description:

        Every year (for 3 years) we decide whether to fish in a lake or not.
        If we fish, the population of fish in the lake remains the same.
        If we don’t fish, the population of fish in the lake doubles.
        Years of experience tell us that our profit is exactly 70% of the
        number of fish in the lake if we fish.
        If there are 10 fish in the lake, we make $7.
        The interest rate is 25%
        We need a contingency plan: no matter what situation we find
        ourselves in, we need to have an optimal strategy.
        We could try every strategy, but that quickly gets out of control:
        there are too many possibilities!

    Output:

        {0: {10: {'fish': False, 'value': 32.25600000000001}},
         1: {10: {'fish': False, 'value': 20.160000000000004},
             20: {'fish': False, 'value': 40.32000000000001}},
         2: {10: {'fish': True, 'value': 12.600000000000001},
             20: {'fish': True, 'value': 25.200000000000003},
             40: {'fish': True, 'value': 50.400000000000006}},
         3: {10: {'fish': True, 'value': 7.0},
             20: {'fish': True, 'value': 14.0},
             40: {'fish': True, 'value': 28.0},
             80: {'fish': True, 'value': 56.0}},
         4: {10: {'value': 0},
             20: {'value': 0},
             40: {'value': 0},
             80: {'value': 0},
             160: {'value': 0}}}
        Optimal strategy: do not fish -> do not fish -> fish -> fish
"""

from copy import deepcopy
from pprint import pprint


# Problem parameters
initial_population = 10
growth_factor = 2
profit_rate = 0.7
interest_rate = 0.25
final_value = 0
n_periods = 4


# Initialize data structures to keep track of calculations
tree = {
    0: {
        initial_population: {},
    },
}
max_population = initial_population


# Construct the tree by forward calculation
for i in range(n_periods):
    tree[i+1] = deepcopy(tree[i])
    max_population *= growth_factor
    tree[i+1][max_population] = {}


# Set the values for all nodes in the last period to 0
for key in tree[n_periods].keys():
    tree[n_periods][key]['value'] = 0


# Value the tree by backward induction
for i in reversed(range(n_periods)):
    for key in tree[i].keys():
        fish = profit_rate * key + (1 / (1 + interest_rate)) * tree[i+1][key]['value']
        no_fish = (1 / (1 + interest_rate)) * tree[i+1][key * growth_factor]['value']
        tree[i][key]['value'] = max(fish, no_fish)
        if fish > no_fish:
            tree[i][key]['fish'] = True
        else:
            tree[i][key]['fish'] = False


# Print the decision rule
result = []
optimal_population = initial_population
for i in range(n_periods):
    if tree[i][optimal_population]['fish']:
        result.append('fish')
    else:
        result.append('do not fish')
        optimal_population *= growth_factor

pprint(tree)
print('Optimal strategy: ' + ' -> '.join(result))
