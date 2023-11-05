"""
Assignment #2
CS 7375
Nikhil Pai

Implementing an A* algorithim to solve an 8 puzzle

"""
import heapq
import numpy as np
import random
import time  

import sys
from PyQt5 import QtCore, QtWidgets, QtGui, uic


class Puzzle:
    """Represents a puzzle node
    """
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state  # current state
        self.parent = parent  # node parent
        self.action = action  # action the node is about to take
        self.cost = cost  # cost of action
        self.heuristic = heuristic  # heuristic value from manhattan
        self.priority = cost + heuristic  # estimated total cost

    def __lt__(self, other):
        # overwritting the less than dunder function
        return self.priority < other.priority

def manhattan_distance(state: np.array, goal_state: np.array) -> int:
    """Returns the city-distance

    Args:
        state (array): current state
        goal_state (array): goal state

    Returns:
        int: total distance(error) of all the tiles
    """

    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state[i, j]
            if tile != 0:
                goal_position = np.where(goal_state == tile)
                distance += abs(i - goal_position[0][0]) + \
                    abs(j - goal_position[1][0])
    return distance


def get_possible_actions(state: np.array) -> list:
    """Determines the valid actions that can be taken from a given state

    Args:
        state (np.array): current state

    Returns:
        list: a list of vaild actions (name, calculation)
    """
    possible_actions = []
    i, j = np.where(state == 0)
    i, j = i[0], j[0]

    if i > 0:
        possible_actions.append(('up', (i - 1, j)))
    if i < 2:
        possible_actions.append(('down', (i + 1, j)))
    if j > 0:
        possible_actions.append(('left', (i, j - 1)))
    if j < 2:
        possible_actions.append(('right', (i, j + 1)))

    return possible_actions


def apply_action(state: np.array, action: tuple) -> np.array:
    """Applies a given action to a state and returns the resulting state

    Args:
        state (np.array): current state
        action (tuple): calculations to find new i and j

    Returns:
        np.array: resulting state
    """
    new_state = state.copy()
    i, j = np.where(state == 0)
    i, j = i[0], j[0]
    new_i, new_j = action[1]
    new_state[i, j], new_state[new_i,
                               new_j] = new_state[new_i, new_j], new_state[i, j]
    return new_state


def reconstruct_path(node: Puzzle):
    """Trace back the actions from the goal node to the start node and return the path as a list

    Args:
        node (Puzzle): one node of the solution

    Returns:
        _type_: list of all the actions taken
    """

    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    return path[::-1]


def get_opposite_action(action: str) -> str:
    """Gets the opposite action

    Args:
        action (str): action

    Returns:
        str: opposite action
    """
    if action == 'up':
        return 'down'
    elif action == 'down':
        return 'up'
    elif action == 'left':
        return 'right'
    elif action == 'right':
        return 'left'


def generate_solvable_start_state(goal_state: np.array, num_moves=50) -> np.array:
    """Generates a solvable state state

    Args:
        goal_state (np.array): goal state
        num_moves (int, optional): _description_. Defaults to 50.

    Returns:
        np.array: start state
    """
    current_state = goal_state.copy()
    previous_action = None

    for _ in range(num_moves):
        possible_actions = get_possible_actions(current_state)

        # Remove the action that undoes the previous action
        if previous_action:
            opposite_action = get_opposite_action(previous_action)
            possible_actions = [
                action for action in possible_actions if action[0] != opposite_action]

        # Randomly select and apply an action
        action = random.choice(possible_actions)
        current_state = apply_action(current_state, action)
        previous_action = action[0]

    return current_state


def solve_puzzle(start_state: np.array, goal_state: np.array):
    """Solves 8 puzzle with a star algorithim

    Args:
        start_state (np.array): start state
        goal_state (np.array): goal state

    Returns:
        list: path taken
        int: nodes visited
    """

    start_node = Puzzle(start_state, None, None, 0,
                        manhattan_distance(start_state, goal_state))
    # used to select the node with the lowest priority (lowest cost + heuristic) during the search
    open_list = [start_node]
    # used to store the states that have already been visited to avoid revisiting them
    closed_set = set()
    visited_nodes = 0

    while open_list:
        # continues until no more nodes to explore or a goal state is reached.
        current_node = heapq.heappop(open_list)
        visited_nodes += 1

        if np.array_equal(current_node.state, goal_state):
            return reconstruct_path(current_node), visited_nodes

        closed_set.add(tuple(current_node.state.flatten()))
        possible_actions = get_possible_actions(current_node.state)

        for action in possible_actions:
            new_state = apply_action(current_node.state, action)
            if tuple(new_state.flatten()) not in closed_set:
                new_cost = current_node.cost + 1
                new_heuristic = manhattan_distance(new_state, goal_state)
                new_node = Puzzle(new_state, current_node, action,
                                  new_cost, new_heuristic)  # create new node
                heapq.heappush(open_list, new_node)

    return None, visited_nodes


def generate_table(runs:int):
    """Gather statistics
    """
    num_runs = runs
    total_visited_nodes = 0
    total_run_time = 0
    actual_runs = 0

    while actual_runs < num_runs:
        goal_state = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
        start_state = generate_solvable_start_state(goal_state)

        start_time = time.time()
        path, visited_nodes = solve_puzzle(start_state, goal_state)
        end_time = time.time()

        if path is not None:
            actual_runs += 1
            total_run_time += end_time - start_time
            total_visited_nodes += visited_nodes

    average_visited_nodes = total_visited_nodes / actual_runs
    average_run_time = total_run_time / actual_runs

    print("Statistics:")
    print(f"Number of iterations: {num_runs}")
    print(f"Average number of nodes visited: {average_visited_nodes}")
    print(f"Average run time: {average_run_time} seconds")

    return num_runs, average_visited_nodes, average_run_time


def one_puzzle():
    """Solves one puzzle

    Returns:
        np.array: start state
        np.array: goal state
        list: path taken
    """
    goal_state = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
    start_state = generate_solvable_start_state(goal_state)

    path, visited_nodes = solve_puzzle(start_state, goal_state)
    if path is not None:
        pass
        print("Solution found! Follow these steps:")
        print(start_state)
        for step, action in enumerate(path):
            print(f"Step {step+1}: Move {action}")
        print(goal_state)
        return start_state, goal_state, path
    else:
        return None


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        self.initUI()
        self.show()

    def initUI(self):
        """Initalize UI
        """
        uic.loadUi('8_puzzle.ui', self)
        self.setWindowTitle('8 Puzzle')
        self.labels = [self.label_1,
                       self.label_2,
                       self.label_3,
                       self.label_4,
                       self.label_5,
                       self.label_6,
                       self.label_7,
                       self.label_8,
                       self.label_9, ]

        self.pushButton_forward.clicked.connect(self.go_forward)
        self.pushButton_back.clicked.connect(self.go_back)
        self.pushButton_puzzle.clicked.connect(self.make_puzzle)
        self.pushButton_stats.clicked.connect(self.update_table)

    def make_puzzle(self):
        """Solves one puzzle and stores it's information
        """
        state, goal_state, path = one_puzzle()
        if path is not None:
            self.state = state
            self.goal_state = goal_state
            self.path = path
            self.flat_state = self.state.flatten()

            self.update_labels()

            self.action_history = [self.state.flatten()]
            self.step = 0
            for action in self.path:
                new_state = apply_action(self.state, action)
                self.action_history.append(new_state.flatten())
                self.state = new_state

    def update_labels(self):
        """Updates labels
        """
        for i, label in enumerate(self.labels):
            label.setText(str(self.flat_state[i]))

    def go_forward(self):
        """Moves puzzle forward one step
        """
        if self.step < len(self.action_history)-1:
            self.step += 1
            self.flat_state = self.action_history[self.step]
            self.update_labels()

    def go_back(self):
        """Moves puzzle back one step
        """
        if self.step > 0:
            self.step -= 1
            self.flat_state = self.action_history[self.step]
            self.update_labels()

    def update_table(self):
        """Runs multiple puzzles and report the stats
        """
        num_runs, average_visited_nodes, average_run_time = generate_table(10)
        self.label_runs.setText(str(num_runs))
        self.label_nodes.setText(str(average_visited_nodes))
        self.label_time.setText(str(average_run_time))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())

