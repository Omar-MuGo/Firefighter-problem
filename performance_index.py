"""Performance index

This file implements different performance indexes for evaluating the fitness of the
different heuristics.
"""

from ffp import *
import copy


def k_step_avg_burning(problem, heuristic, k=8):

    problem_copy = copy.deepcopy(problem)

    graph         = problem_copy.graph
    initial_state = problem_copy.state

    sim = FFP_SIM(graph, initial_state)

    # Protect a node based on the heuristic
    saved_node           = sim.nextNode(heuristic)
    sim.state[saved_node] = 1

    average_burning_nodes = sim.avgBurnNodesSimK(k)

    return average_burning_nodes
