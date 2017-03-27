"""
    rolling_dice.py
    ~~~~~~~~~~~~~~~

    This module solves the rolling dice in-class problem.

    Run with "python rolling_dice.py"
"""


def average(i, outcomes):
    rsp = 0
    for j in range(1, 7):
        rsp += outcomes.get((i, j), 0)
    return rsp / 6


def simulate(n_rounds):
    outcomes = {}
    for i in reversed(range(1, n_rounds + 1)):
        for j in range(1, 7):
            outcomes[(i, j)] = max(j, average(i + 1, outcomes))
    return outcomes


i = 0
while  True:
    i += 1
    outcomes = simulate(i)
    below_5 = False
    for j in range(1, 7):
        if outcomes[(1, j)] < 5:
            below_5 = True

    if not below_5:
        print('rounds needed: {}'.format(i))
        break
