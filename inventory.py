"""
inventory.py
~~~~~~~~~~~~

This module solves an inventory decision problem.

Run with "python inventory.py"

Problem description:

    "Suppose I sell iPads. I plan on selling iPads for 10 weeks. Every week,
    I place an order with Apple and the iPads arrive instantly. If I run out
    of inventory, I agree to give customers a “rain check.” This means
    items are backordered."
"""

import random

import numpy as np

# Use the tqdm package to show a nice status bar.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class InventoryDP:
    """A dynamic programming solution to an inventory decision problem.

    Parameters
    ----------
    time_horizon : integer
        Number of time periods to be simulated in the decision tree
    holding_cost : float
        Cost in Dollars to keep one item on stock for one period
    price : float
        Sales price per item used to model "rain check" costs
    max_inventory : integer
        Maximum number of units that can be kept on stock at any time
    max_demand : integer
        Maximum number of units that can be demanded in a time period

    Returns
    -------
    self : InventoryDP
        The instance is returned to allow method chaining.
    """

    def __init__(self, time_horizon=10, holding_cost=5, price=100,
                 max_inventory=100, max_demand=10):
        self.time_horizon = time_horizon
        self.holding_cost = holding_cost
        self.price = price
        self.max_inventory = max_inventory
        self.max_demand = max_demand
        self._tree_height = None

    def _initialize(self):
        """Initialize the data structures that represent the decision tree"""

        # The decision tree is modeled implicitly by storing the nodes' data
        # in two two-dimensional arrays of appropriate sizes. The first
        # dimension represents the inventory level. Since the problem
        # formulation allows for negative inventories (= backorders), the
        # "center" index in the first dimension equals a zero inventory level.
        self._tree_height = 2 * self.max_inventory + 1
        self._shape = (self._tree_height, self.time_horizon + 1)
        self._values = np.zeros(self._shape, dtype=np.float64)
        self._decisions = np.zeros(self._shape, dtype=np.float64)

    def recurse(self):
        """Built the decision tree and solve it by backwards induction"""

        self._initialize()

        # The values in the terminal nodes are assumed to be 0.
        # Since the values array is initialized to zeroes,
        # nothing needs to be computed for the terminal nodes.

        # Initialize the progress bar.
        if tqdm:
            total = self.time_horizon * self._tree_height
            pbar = tqdm(total=total)

        # Iterate chronologically backwards over time and over all states.
        # This implicitly visits all of the tree's nodes.
        for period in reversed(range(self.time_horizon)):
            for state in range(self._tree_height):

                # Update the progress bar.
                if tqdm:
                    pbar.update()

                # Since the state dimension of the decision tree includes
                # negative values for the inventory (= backorders), the
                # index variable needs to be adjusted.
                current_inventory = state - self.max_inventory

                # Create a two-dimensional array to store the temporary values
                # that go into the expexted value calculations further below.
                # The first dimension is of a size equal to the potential
                # demands whereas the second dimension is equal to the number
                # of maximum possible order size (i.e., the maximum number such
                # that the inventory does not exceed max_inventory in the edge
                # case of zero demand). For example, when max_inventory = 100,
                # current_inventory = 95, and max_demand = 10, the decision
                # maker could order 15 items at most (if there is no demand).
                # Since it is assumed that the demand realization occurs
                # after the decision maker places and receives the order, the
                # potential excess of inventory over max_inventory will be
                # forfeited.
                possible_orders = (self.max_inventory - current_inventory
                                   + self.max_demand)
                shape = (self.max_demand + 1, possible_orders)
                local_values = np.zeros(shape, dtype=np.float64)

                # Iterate over the cartesian product of possible orders and
                # demand realizations and update the state variables.
                for order in range(possible_orders):
                    for demand in range(self.max_demand + 1):
                        
                        # Calculate the inventory for the consequent period.
                        new_inventory = current_inventory + order - demand

                        # If the calculated inventory is above or below
                        # the max_inventory, the value is clipped and a
                        # penalty for either loosing a customer entirely
                        # or not being able to keep the excess on stock
                        # is introduced.
                        if new_inventory < -1 * self.max_inventory:
                            new_inventory = -1 * self.max_inventory
                            throw_away_penalty = self.price * 2
                        elif new_inventory > self.max_inventory:
                            new_inventory = self.max_inventory
                            throw_away_penalty = self.price * 2
                        else:
                            throw_away_penalty = 0

                        # Calculate the realized cost.
                        local_values[demand, order] = (
                            self.holding_cost * max(0, new_inventory)
                            + self.price * max(0, -1 * new_inventory)
                            + self._values[new_inventory + self.max_inventory,
                                           period + 1]
                            + throw_away_penalty
                        )

                # Calculate the expected value for each possible decision.
                expected_values = (local_values.sum(axis=0) / (self.max_demand + 1))

                # Save the minimum expected value in the tree matrices.
                self._values[state, period] = expected_values.min()
                self._decisions[state, period] = expected_values.argmin()

        if tqdm:
            pbar.close()

        return self

    def simulate(self, initial_inventory):
        """Simulate the decision problem and calculate the actual costs.

        Parameters
        ----------
        initial_inventory : integer
            Level of inventory at the beginning of the time horizon.
            Must be between -max_inventory and +max_inventory.
            If negative, the decision problem starts with a backorder.

        Returns
        -------
        (costs, orders, demands) : float, ndarray [time_horizon],
                                   ndarray [time_horizon]
            The return value is a tuple that consists of the total
            costs and two arrays with the orders and realized
            demands for each period.

        Raises
        ------
        ValueError
            If initial_inventory is not between -max_inventory and
            +max_inventory.
        RuntimeError
            If InventoryDP.recurse() was not run before.
        """

        # Check the initial_inventory parameter.
        if not (0 <= initial_inventory <= abs(self.max_inventory)):
            raise ValueError('initial_inventory out of range')

        # Ensure that the dynamic program was run before.
        if self._tree_height is None:
            raise RuntimeError('Call InventoryDP.recurse() first')

        # Initialize the data structures for the final output.
        shape = (self.time_horizon, )
        orders = np.zeros(shape, dtype=np.int32)
        demands = np.zeros(shape, dtype=np.int32)
        costs = 0

        # Iterate over the time horizon in chronological order.
        for period in range(self.time_horizon):

            # Determine the optimal order quantity.
            state = initial_inventory + self.max_inventory
            orders[period] = self._decisions[state, period]

            # Get a random demand realization.
            demands[period] = random.randint(0, self.max_demand)

            # Calculate the inventory for the consequent period.
            initial_inventory += orders[period] - demands[period]

            # If the calculated inventory is above or below
            # the max_inventory, the value is clipped and a
            # penalty for either loosing a customer entirely
            # or not being able to keep the excess on stock
            # is introduced.
            if initial_inventory < -1 * self.max_inventory:
                initial_inventory = -1 * self.max_inventory
                throw_away_penalty = self.price * 2
            elif initial_inventory > self.max_inventory:
                initial_inventory = self.max_inventory
                throw_away_penalty = self.price * 2
            else:
                throw_away_penalty = 0

            # Accumulate the total incurred costs.
            costs += (
                self.holding_cost * max(0, initial_inventory)
                + self.price * max(0, -1 * initial_inventory)
                + throw_away_penalty
            )

        return costs, orders, demands


# Create a list of experiments with varied parameter settings.
experiments = [
    {
        'time_horizon': 10,
        'holding_cost': 5,
        'price': 5,
        'max_inventory': 100,
        'max_demand': 10,
    },
    {
        'time_horizon': 10,
        'holding_cost': 10,
        'price': 5,
        'max_inventory': 100,
        'max_demand': 10,
    },
    {
        'time_horizon': 10,
        'holding_cost': 5,
        'price': 10,
        'max_inventory': 100,
        'max_demand': 10,
    },
]


# Run the experiments.
for i, parameters in enumerate(experiments):
    print()
    print('Experiment #{:d}'.format(i+1))
    print('==============')
    print()
    print(' Parameters: ', parameters)
    print()

    dp = InventoryDP(**parameters).recurse()

    # Simulate two independent runs for each experiment.
    for j in range(2):
        costs, orders, demands = dp.simulate(initial_inventory=10)

        print()
        print(' Simulation #{:d}'.format(j+1))
        for k in range(len(orders)):
            print('   Period #{:d}: {:d} ordered and {:d} sold'
                  .format(k+1, orders[k], demands[k]))
        print('  Total costs: {:.2f}'.format(costs))
