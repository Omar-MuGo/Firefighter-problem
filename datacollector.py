"""DataCollector

This file contains the Data Collector class definition.
DataCollector allows us to obtain the features from all problem instances to be able to
create a classifier based on such features.
"""
import pandas as pd


class DataCollector:
    def __init__(self):
        # Create class variables
        self.database = pd.DataFrame()

    def extract_problem_features(self, problem, heuristics, performance_index):
        valid_features = ["EDGE_DENSITY", "AVG_DEGREE", "BURNING_NODES", "BURNING_EDGES", "NODES_IN_DANGER"]

        problem_features = {}
        heuristics_performance = {}

        # Extract the graph features
        for feature in valid_features:
            problem_features[feature] = [problem.getFeature(feature)]

        # Evaluate the performance index of all heuristics
        for heuristic in heuristics:
            heuristics_performance[heuristic] = performance_index(problem, heuristic)

        # Use the minimum of the performance indexes as the class
        problem_class = min(heuristics_performance, key=heuristics_performance.get)

        # Add the problem class to the database
        problem_features["CLASS"] = [problem_class]

        problem_feat_df = pd.DataFrame(problem_features)

        # Add the problem features to the database
        self.database = pd.concat([self.database, problem_feat_df], ignore_index=True)
