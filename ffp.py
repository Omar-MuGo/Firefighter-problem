"""Firefighter problem

This file contains the source code to run the firefighter problem
"""

import random
from hyperheuristic import HyperHeuristic, ClassifierHyperHeuristic
from datacollector import DataCollector
from performance_index import *
import copy
import pandas as pd
import glob
import os
from instances import instance_manager
import tqdm

# Provides the methods to create and solve the firefighter problem
class FFP:

  # Constructor
  #   fileName = The name of the file that contains the FFP instance
  def __init__(self, fileName):
    file = open(fileName, "r")
    text = file.read()
    tokens = text.split()
    seed = int(tokens.pop(0))
    self.n = int(tokens.pop(0))
    model = int(tokens.pop(0))
    int(tokens.pop(0)) # Ignored
    # self.state contains the state of each node
    #    -1 On fire
    #     0 Available for analysis
    #     1 Protected
    self.state = [0] * self.n
    nbBurning = int(tokens.pop(0))
    for i in range(nbBurning):
      b = int(tokens.pop(0))
      self.state[b] = -1
    self.graph = []
    for i in range(self.n):
      self.graph.append([0] * self.n);
    while tokens:
      x = int(tokens.pop(0))
      y = int(tokens.pop(0))
      self.graph[x][y] = 1
      self.graph[y][x] = 1

  # Solves the FFP by using a given method and a number of firefighters
  #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
  #   nbFighters = The number of available firefighters per turn
  #   debug = A flag to indicate if debugging messages are shown or not
  def solve(self, method, nbFighters, debug = False):
    spreading = True
    if (debug):
      print("Initial state:" + str(self.state))
    t = 0
    while (spreading):
      if (debug):
        print("Features")
        print("")
        print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
        print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
        print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
        print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
        print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))

      #######################################################################
      # PROTECTION STAGE
      #######################################################################
      # It protects the nodes (based on the number of available firefighters)
      for i in range(nbFighters):
        heuristic = method
        if (isinstance(method, HyperHeuristic)):
          heuristic = method.nextHeuristic(self)
        node = self.nextNode(heuristic)
        if (node >= 0):
          # The node is protected
          self.state[node] = 1
          # The node is disconnected from the rest of the graph
          for j in range(len(self.graph[node])):
            self.graph[node][j] = 0
            self.graph[j][node] = 0
          if (debug):
            print("\tt" + str(t) + ": A firefighter protects node " + str(node))

      #######################################################################
      # SPREADING STAGE
      #######################################################################
      # It spreads the fire among the unprotected nodes
      spreading = False
      state = self.state.copy()
      for i in range(len(state)):
        # If the node is on fire, the fire propagates among its neighbors
        if (state[i] == -1):
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and state[j] == 0):
              spreading = True
              # The neighbor is also on fire
              self.state[j] = -1
              # The edge between the nodes is removed (it will no longer be used)
              self.graph[i][j] = 0
              self.graph[j][i] = 0
              if (debug):
                print("\tt" + str(t) + ": Fire spreads to node " + str(j))
      t = t + 1
      if (debug):
        print("---------------")
    if (debug):
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    return self.getFeature("BURNING_NODES")

  # Selects the next node to protect by a firefighter
  #   heuristic = A string with the name of one available heuristic
  def nextNode(self, heuristic):
    index  = -1
    best = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        index = i
        break
    value = -1

    # Loop through the graph
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        if (heuristic == "LDEG"):
          # It prefers the node with the largest degree, but it only considers
          # the nodes directly connected to a node on fire
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and self.state[j] == -1):
              value = sum(self.graph[i])
              break
        elif (heuristic == "GDEG"):
          value = sum(self.graph[i])
        else:
          print("=====================")
          print("Critical error at FFP.nextNode.")
          print("Heuristic " + heuristic + " is not recognized by the system.")
          print("The system will halt.")
          print("=====================")
          exit(0)
      if (value > best):
        best = value
        index = i
    return index

  # Returns the value of the feature provided as argument
  #   feature = A string with the name of one available feature
  def getFeature(self, feature):
    f = 0
    if (feature == "EDGE_DENSITY"):
      n = len(self.graph)
      for i in range(len(self.graph)):
        f = f + sum(self.graph[i])
      f = f / (n * (n - 1))
    elif (feature == "AVG_DEGREE"):
      n = len(self.graph)
      count = 0
      for i in range(len(self.state)):
        if (self.state[i] == 0):
          f += sum(self.graph[i])
          count += 1
      if (count > 0):
        f /= count
        f /= (n - 1)
      else:
        f = 0
    elif (feature == "BURNING_NODES"):
      for i in range(len(self.state)):
        if (self.state[i] == -1):
          f += 1
      f = f / len(self.state)
    elif (feature == "BURNING_EDGES"):
      n = len(self.graph)
      for i in range(len(self.graph)):
        for j in range(len(self.graph[i])):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
      f = f / (n * (n - 1))
    elif  (feature == "NODES_IN_DANGER"):
      for j in range(len(self.state)):
        for i in range(len(self.state)):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
            break
      f /= len(self.state)
    else:
      print("=====================")
      print("Critical error at FFP._getFeature.")
      print("Feature " + feature + " is not recognized by the system.")
      print("The system will halt.")
      print("=====================")
      exit(0)
    return f

  # Returns the string representation of this problem
  def __str__(self):
    text = "n = " + str(self.n) + "\n"
    text += "state = " + str(self.state) + "\n"
    for i in range(self.n):
      for j in range(self.n):
        if (self.graph[i][j] == 1 and i < j):
          text += "\t" + str(i) + " - " + str(j) + "\n"
    return text

class FFP_SIM(FFP):
  def __init__(self, graph, state):
    self.graph = graph
    self.state = state

  def solve(self, debug = False):
    '''
    The function simulates one cycle of the firefighter problem
    :param debug:
    :return:
    '''

    if (debug):
      print("Initial state:" + str(self.state))
      print("Features")
      print("")
      print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
      print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
      print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
      print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
      print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))

    #######################################################################
    # SPREADING STAGE
    #######################################################################
    # It spreads the fire among the unprotected nodes
    state = self.state.copy()
    for i in range(len(state)):
      # If the node is on fire, the fire propagates among its neighbors
      if (state[i] == -1):
        for j in range(len(self.graph[i])):
          if (self.graph[i][j] == 1 and state[j] == 0):
            # The neighbor is also on fire
            self.state[j] = -1
            # The edge between the nodes is removed (it will no longer be used)
            self.graph[i][j] = 0
            self.graph[j][i] = 0
            if (debug):
              print("\t Fire spreads to node " + str(j))

    if (debug):
      print("---------------")

    if (debug):
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    return self.getFeature("BURNING_NODES")

  def avgBurnNodesSimK(self, k):
    '''
    Simulates the system K steps into the future and computes the average number of nodes burnt.
    :param k:
    :return:
    '''

    sum_burning_nodes = 0

    for i in range(k):
      burning_nodes = self.solve()
      sum_burning_nodes += burning_nodes

    average_burning_nodes = sum_burning_nodes / k

    return average_burning_nodes

def collect_problem_data(data_collector, problem, heuristics, performance_index):
  ALL_NODES_BURNT = 1.0

  problem_copy = copy.deepcopy(problem)

  graph = problem_copy.graph
  initial_state = problem_copy.state

  sim = FFP_SIM(graph, initial_state)

  burning_nodes   = sim.getFeature("BURNING_NODES")
  while burning_nodes < ALL_NODES_BURNT:
    # Collect problem data
    data_collector.extract_problem_features(sim, heuristics, performance_index)

    # Evolve the problem 1 timestep
    burning_nodes = sim.solve()

def create_database(folder, heuristics, performance_index):
  max_iter        = 3
  enable_max_iter = False

  # Collect data from all problem instances
  # files = glob.glob(folder + '/**/*.in', recursive=True)
  files = instance_manager.get_training_filenames()

  # Testing the data collection process
  data_collector = DataCollector()

  i = 0
  for fileName in tqdm.tqdm(files):
    # Break when max iteration is reached
    if i >= max_iter:
      break

    # Solves the problem using heuristic LDEG and one firefighter
    problem = FFP(fileName)
    collect_problem_data(data_collector, problem, heuristics, k_step_avg_burning)

    # Increase iteration only for valid max_iter param
    if enable_max_iter:
      i += 1

  pd.set_option('display.max_columns', None)
  print(data_collector.database.head())

  return data_collector.database

def main():
  # Parameter data
  DB_CREATE = False
  ET_CREATE = False

  # Local variables
  heuristics = ["LDEG", "GDEG"]

  if DB_CREATE:

    # Collect data from all problem instances
    input_folder = "instances/01_training/"
    database = create_database(input_folder, heuristics, k_step_avg_burning)

    # Save database to file
    output_folder = "database/"

    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    database.to_pickle(output_folder + 'database.pkl')

  ######################################################
  # Quick test of HH vs H1 vs H2
  ######################################################
  # filename = "instances/01_testing/1000_r0.05_0_geom_8.gin"

  # # Solve using LDEG heuristic
  # problem = FFP(filename)
  # print("LDEG = " + str(problem.solve("LDEG", 1, False)))

  # # Solve using GDEG heuristic
  # problem = FFP(filename)
  # print("GDEG = " + str(problem.solve("GDEG", 1, False)))

  # # Solve using HH
  # problem    = FFP(filename)
  # model_file = 'models/dt_classifier_hh.pkl'

  # features = ["EDGE_DENSITY", "AVG_DEGREE", "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
  # hh = ClassifierHyperHeuristic(features, model_file)
  # print("Classifier HH = " + str(problem.solve(hh, 1, False)))

  if ET_CREATE == True:

    features = ["EDGE_DENSITY", "AVG_DEGREE", "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]
    model_file = 'models/dt_classifier_hh.pkl'

    # Generate evaluation table
    models = {
      "GDEG":"GDEG",
      "LDEG":"LDEG",
      "Classifier HH":ClassifierHyperHeuristic(features, model_file)
    }
    instance_manager.get_evaluation_table(models)

if __name__ == "__main__":
  main()
