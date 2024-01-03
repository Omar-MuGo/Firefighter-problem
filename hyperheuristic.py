"""Hyperheuristic

This file contains the definition of the different Hyperheuristics used to solve the
firefighter problem.
"""

import random
import math
import pickle

# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation
import pandas as pd


class HyperHeuristic:

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    def __init__(self, features, heuristics):
        if (features):
            self.features = features.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of features cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)
        if (heuristics):
            self.heuristics = heuristics.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of heuristics cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        print("=====================")
        print("Critical error at HyperHeuristic.nextHeuristic.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)

    # Returns the string representation of this hyper-heuristic
    def __str__(self):
        print("=====================")
        print("Critical error at HyperHeuristic.__str__.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)


# A dummy hyper-heuristic for testing purposes.
# The hyper-heuristic creates a set of randomly initialized rules.
# Then, when called, it measures the distance between the current state and the
# conditions in the rules
# The rule with the condition closest to the problem state is the one that fires
class DummyHyperHeuristic(HyperHeuristic):

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   nbRules = The number of rules to be contained in this hyper-heuristic
    def __init__(self, features, heuristics, nbRules, seed):
        super().__init__(features, heuristics)
        random.seed(seed)
        self.conditions = []
        self.actions = []
        for i in range(nbRules):
            self.conditions.append([0] * len(features))
            for j in range(len(features)):
                self.conditions[i][j] = random.random()
            self.actions.append(heuristics[random.randint(0, len(heuristics) - 1)])

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        minDistance = float("inf")
        index = -1
        state = []
        for i in range(len(self.features)):
            state.append(problem.getFeature(self.features[i]))
        print("\t" + str(state))
        for i in range(len(self.conditions)):
            distance = self.__distance(self.conditions[i], state)
            if (distance < minDistance):
                minDistance = distance
                index = i
        heuristic = self.actions[index]
        print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
        return heuristic

    # Returns the string representation of this dummy hyper-heuristic
    def __str__(self):
        text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
        for i in range(len(self.conditions)):
            text += "\t" + str(self.conditions[i]) + " => " + self.actions[i] + "\n"
        return text

    # Returns the Euclidian distance between two vectors
    def __distance(self, vectorA, vectorB):
        distance = 0
        for i in range(len(vectorA)):
            distance += (vectorA[i] - vectorB[i]) ** 2
        distance = math.sqrt(distance)
        return distance

class ClassifierHyperHeuristic(HyperHeuristic):
    def __init__(self, features, model_file):
        self.features = features

        # Load model from filename
        with open(model_file, 'rb') as file:
            model = pickle.load(file)

        self.model = model


    def nextHeuristic(self, problem):
        state = []

		# Compute the features
        for i in range(len(self.features)):
            state.append(problem.getFeature(self.features[i]))

		# Make a prediction
        state_df         = pd.DataFrame([state], columns=self.features)
        heuristic_val    = self.model.predict(state_df)
        heuristic_val    = heuristic_val[0]
        heuristic_string = self.mapHeuristicVal2String(heuristic_val)

        return heuristic_string

    def mapHeuristicVal2String(self, heuristic_val):
        heuristic_dict = {0:"GDEG", 1:"LDEG"}

        return heuristic_dict.get(heuristic_val)