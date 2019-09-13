# coding: utf-8
import os
import pandas as pd
import numpy as np
import tempfile
import warnings
import zipfile
from shutil import copyfile
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from werkzeug.utils import secure_filename

BEGIN_TIME = float(os.getenv("BEGIN_TIME", 1568382895.523836))


class NovartisHackathonEvaluator:
    def __init__(self, ground_truth_folder, debug=False):
        """
        Tasks :

        - Task-1 : Submit As Fast as your can || TODO
        - Task-2 : Identifying the Primary Key
        - Task-3 : Calculating the Silhouette Score
        - Task-4 : Calculating the MAPE
        - Task-5 : Calculating the score for optimization
        """
        self.ground_truth_folder = ground_truth_folder
        self.debug = debug
        self.setup_ground_truths()

        self.task_scoring_weights = {
            1: 0.1,
            2: 0.1,
            3: 0.3,
            4: 0.35,
            5: 0.15
        }

    def setup_ground_truths(self):
        """
        Helper function to setup Ground Truths for all the individual
        Tasks
        """
        self.task2_ground_truth_path = os.path.join(
            self.ground_truth_folder,
            "task2_ground_truth.csv"
        )

        self.task4_ground_truth_path = os.path.join(
            self.ground_truth_folder,
            "task4_ground_truth.csv"
        )

        self.task5_ground_truth_path = os.path.join(
            self.ground_truth_folder,
            "task5_ground_truth.csv"
        )

    def compute_task1_score(self):
        """
        We use a constant value of 1 for now
        """
        return 1

    def compute_task2_score(self, task2_submission_path):
        """

        Params:
            - task2_submission_path
                Path to the CSV file with primary key identifications,
                submitted by the participant
        """

        # Setup DataFrames
        participant_submission = pd.read_csv(
            task2_submission_path
        )
        ground_truth = pd.read_csv(
            self.task2_ground_truth_path
        )

        # Validate Submission File
        for expected_column in ["table_name", "primary_key"]:
            assert expected_column in participant_submission.columns,\
                "{} not present in Task-2 submission header".format(
                    expected_column
                )

        # Compute Compound Column for both ground truth and participant
        # submission
        for df in [ground_truth, participant_submission]:
            df["compound_column"] = \
                df["table_name"] + "--" + df["primary_key"]

        expected_values = set(ground_truth["compound_column"].tolist())
        submitted_values = set(
            participant_submission["compound_column"].tolist())

        mismatches = expected_values - submitted_values

        if len(mismatches) != 0:
            print(
                "The participant needs to rework on the solution. There are {} lines missing. ".format(
                    len(mismatches)))

        """
        A score of [0, 1] where
            - 1 refers to a perfect match
            - 0 refers to a perfect mismatch
        """
        score = 1 - (len(mismatches) / len(expected_values))
        return score

    def compute_task3_score(self, task3_submission_path):
        """

        Params:
            - task3_submission_path
                Path to the CSV dataset shared by the participant that
                has the predicted value in the "Market_Cluster" column
                for each "Hospital_ID" along with all features used
                for clustering
        """

        # Setup DataFrames
        participant_submission = pd.read_csv(
            task3_submission_path
        )

        # Validate Submission File
        expected_columns = "Hospital_ID,Market_Cluster,Population,Prevalence,Total_HCPs,A_Sales_201804,A_Sales_201805,A_Sales_201806,A_Sales_201807,A_Sales_201808,A_Sales_201809,A_Sales_201810,A_Sales_201811,A_Sales_201812,A_Sales_201901,A_Sales_201902,A_Sales_201903,A_Calls_201804,A_Calls_201805,A_Calls_201806,A_Calls_201807,A_Calls_201808,A_Calls_201809,A_Calls_201810,A_Calls_201811,A_Calls_201812,A_Calls_201901,A_Calls_201902,A_Calls_201903,A_Events_201804,A_Events_201805,A_Events_201806,A_Events_201807,A_Events_201808,A_Events_201809,A_Events_201810,A_Events_201811,A_Events_201812,A_Events_201901,A_Events_201902,A_Events_201903,A_Web_Visits_201804,A_Web_Visits_201805,A_Web_Visits_201806,A_Web_Visits_201807,A_Web_Visits_201808,A_Web_Visits_201809,A_Web_Visits_201810,A_Web_Visits_201811,A_Web_Visits_201812,A_Web_Visits_201901,A_Web_Visits_201902,A_Web_Visits_201903,A_Webinars_201804,A_Webinars_201805,A_Webinars_201806,A_Webinars_201807,A_Webinars_201808,A_Webinars_201809,A_Webinars_201810,A_Webinars_201811,A_Webinars_201812,A_Webinars_201901,A_Webinars_201902,A_Webinars_201903,MWC17_Unq_Views_201804,MWC17_Unq_Views_201805,MWC17_Unq_Views_201806,MWC17_Unq_Views_201807,MWC17_Unq_Views_201808,MWC17_Unq_Views_201809,MWC17_Unq_Views_201810,MWC17_Unq_Views_201811,MWC17_Unq_Views_201812,MWC17_Unq_Views_201901,MWC17_Unq_Views_201902,MWC17_Unq_Views_201903,MWC17_Unq_Clicks_201804,MWC17_Unq_Clicks_201805,MWC17_Unq_Clicks_201806,MWC17_Unq_Clicks_201807,MWC17_Unq_Clicks_201808,MWC17_Unq_Clicks_201809,MWC17_Unq_Clicks_201810,MWC17_Unq_Clicks_201811,MWC17_Unq_Clicks_201812,MWC17_Unq_Clicks_201901,MWC17_Unq_Clicks_201902,MWC17_Unq_Clicks_201903,ESMO17_Unq_Views_201804,ESMO17_Unq_Views_201805,ESMO17_Unq_Views_201806,ESMO17_Unq_Views_201807,ESMO17_Unq_Views_201808,ESMO17_Unq_Views_201809,ESMO17_Unq_Views_201810,ESMO17_Unq_Views_201811,ESMO17_Unq_Views_201812,ESMO17_Unq_Views_201901,ESMO17_Unq_Views_201902,ESMO17_Unq_Views_201903,ESMO17_Unq_Clicks_201804,ESMO17_Unq_Clicks_201805,ESMO17_Unq_Clicks_201806,ESMO17_Unq_Clicks_201807,ESMO17_Unq_Clicks_201808,ESMO17_Unq_Clicks_201809,ESMO17_Unq_Clicks_201810,ESMO17_Unq_Clicks_201811,ESMO17_Unq_Clicks_201812,ESMO17_Unq_Clicks_201901,ESMO17_Unq_Clicks_201902,ESMO17_Unq_Clicks_201903,ESCO17_Unq_Opens_201804,ESCO17_Unq_Opens_201805,ESCO17_Unq_Opens_201806,ESCO17_Unq_Opens_201807,ESCO17_Unq_Opens_201808,ESCO17_Unq_Opens_201809,ESCO17_Unq_Opens_201810,ESCO17_Unq_Opens_201811,ESCO17_Unq_Opens_201812,ESCO17_Unq_Opens_201901,ESCO17_Unq_Opens_201902,ESCO17_Unq_Opens_201903,ESCO17_Unq_Clicks_201804,ESCO17_Unq_Clicks_201805,ESCO17_Unq_Clicks_201806,ESCO17_Unq_Clicks_201807,ESCO17_Unq_Clicks_201808,ESCO17_Unq_Clicks_201809,ESCO17_Unq_Clicks_201810,ESCO17_Unq_Clicks_201811,ESCO17_Unq_Clicks_201812,ESCO17_Unq_Clicks_201901,ESCO17_Unq_Clicks_201902,ESCO17_Unq_Clicks_201903,ASCO18_Unq_Views_201804,ASCO18_Unq_Views_201805,ASCO18_Unq_Views_201806,ASCO18_Unq_Views_201807,ASCO18_Unq_Views_201808,ASCO18_Unq_Views_201809,ASCO18_Unq_Views_201810,ASCO18_Unq_Views_201811,ASCO18_Unq_Views_201812,ASCO18_Unq_Views_201901,ASCO18_Unq_Views_201902,ASCO18_Unq_Views_201903,ASCO18_Unq_Clicks_201804,ASCO18_Unq_Clicks_201805,ASCO18_Unq_Clicks_201806,ASCO18_Unq_Clicks_201807,ASCO18_Unq_Clicks_201808,ASCO18_Unq_Clicks_201809,ASCO18_Unq_Clicks_201810,ASCO18_Unq_Clicks_201811,ASCO18_Unq_Clicks_201812,ASCO18_Unq_Clicks_201901,ASCO18_Unq_Clicks_201902,ASCO18_Unq_Clicks_201903,EBCC18_Unq_Views_201804,EBCC18_Unq_Views_201805,EBCC18_Unq_Views_201806,EBCC18_Unq_Views_201807,EBCC18_Unq_Views_201808,EBCC18_Unq_Views_201809,EBCC18_Unq_Views_201810,EBCC18_Unq_Views_201811,EBCC18_Unq_Views_201812,EBCC18_Unq_Views_201901,EBCC18_Unq_Views_201902,EBCC18_Unq_Views_201903,EBCC18_Unq_Clicks_201804,EBCC18_Unq_Clicks_201805,EBCC18_Unq_Clicks_201806,EBCC18_Unq_Clicks_201807,EBCC18_Unq_Clicks_201808,EBCC18_Unq_Clicks_201809,EBCC18_Unq_Clicks_201810,EBCC18_Unq_Clicks_201811,EBCC18_Unq_Clicks_201812,EBCC18_Unq_Clicks_201901,EBCC18_Unq_Clicks_201902,EBCC18_Unq_Clicks_201903"  # noqa
        expected_columns = expected_columns.split(",")
        for expected_column in expected_columns:
            assert expected_column in participant_submission.columns,\
                "{} not present in Task-3 submission header".format(
                    expected_column
                )

        """
        NOTE: You can add more validations, and simply report errors back to the 
        user by doing :
            raise Exception("your error message here")

        Having a good amount of useful validations will help participants 
        better interact with the whole setup, and will save everyone a lot 
        of time in terms of support.
        """

        # Extracting the number of clusters
        n_cluster = participant_submission['Market_Cluster'].max()
        labels = participant_submission['Market_Cluster'].values

        # Dropping the hospital_id column
        df_cluster = participant_submission.drop(
            ['Hospital_ID', 'Market_Cluster'], axis=1)

        # Standardizing the dataframe
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(df_cluster)

        # Silhouette Score
        score = silhouette_score(X, labels, metric='euclidean')
        """
        silhouetter score is [-1, 1]
        so we scale it to [0, 1]
        """
        score = (score + 1) / 2
        return score

    def compute_task4_score(self, task4_submission_path):
        """

        Params:
            - task4_submission_path
                Path to the CSV shared by the participant
                containing the predicted value of sales for the
                provided list of hospitals
        """

        # Setup DataFrames
        participant_submission = pd.read_csv(
            task4_submission_path
        )
        ground_truth = pd.read_csv(
            self.task4_ground_truth_path
        )

        # Validate Submission File
        expected_columns = ['Hospital_ID', 'Month', 'Predict_Sales']
        for expected_column in expected_columns:
            assert expected_column in participant_submission.columns,\
                "{} not present in Task-4 submission header".format(
                    expected_column
                )

        # merging dataframes
        df_mape = pd.merge(
            participant_submission, ground_truth,
            on=['Hospital_ID', 'Month'], how='outer')
        df_mape = df_mape.fillna(0)

        # MAPE
        df_mape['ABS'] = (df_mape['Actual_Sales'] -
                          df_mape['Predict_Sales']).abs() / df_mape['Actual_Sales']
        MAPE = df_mape['ABS'].mean()

        MAPA = np.clip((1 - MAPE), 0, 1)
        """
        Normalization Strategy :
            - Clip MAPA to [0, 1]
        
        Considerations:
            - MAPE is assymetric anyway.
            - Probably test the data distribution for estimation of probability 
                of higher-estimations (which MAPE exagerates by design)
        """

        return MAPA

    def compute_task5_score(self, task5_submission_path, MAPA):
        """
        This function generates the score for the overall optimized value

        Params:
            - task5_submission_path
                Path to the CSV shared by the participant CSV
                that contains the optimized Sales for 12 months
                for all hospitals
        """
        # Setup DataFrames
        participant_submission = pd.read_csv(
            task5_submission_path,
            header=None
        ).dropna()

        ground_truth = pd.read_csv(
            self.task5_ground_truth_path,
            header=None
        ).dropna()

        submission_value = participant_submission.values
        submission_scalar = np.asscalar(submission_value)

        actual_value = ground_truth.values
        actual_scalar = np.asscalar(actual_value)

        # Error Score
        score = (submission_scalar * MAPA) / actual_scalar
        """
        TODO : Normalize these scores to fall within a particular range

        Another key problem is that if participants submit arbitrarily 
        large values, lets say 1000000000 for the submission_value
        then they have can get arbitrarily large scores
        So some form of clipping has to be done here.
        """

        return score

    def evaluate(self, submission_folder):

        # Round-1 Score 
        score_1 = self.compute_task1_score()
        score_1_weighted = score_1 * self.task_scoring_weights[1]

        score_2 = self.compute_task2_score(os.path.join(
            submission_folder,
            "task2_submission.csv"
        ))
        score_2_weighted = score_2 * self.task_scoring_weights[2]

        if self.debug:
            print("Round-2 Score ", score_2)
            print("Round-2 Weighted Score ", score_2_weighted)

        score_3 = self.compute_task3_score(os.path.join(
            submission_folder,
            "task3_submission.csv"
        ))
        score_3_weighted = score_3 * self.task_scoring_weights[3]

        if self.debug:
            print("Round-3 Score ", score_3)
            print("Round-3 Weighted Score ", score_3_weighted)

        score_4 = self.compute_task4_score(os.path.join(
            submission_folder,
            "task4_submission.csv"
        ))
        score_4_weighted = score_4 * self.task_scoring_weights[4]
        # Task-4 Score is the MAPA
        MAPA = score_4

        if self.debug:
            print("Round-4 Score ", score_4)
            print("Round-4 Weighted Score ", score_4_weighted)

        score_5 = self.compute_task5_score(os.path.join(
            submission_folder,
            "task5_submission.csv"
        ), MAPA)
        score_5_weighted = score_5 * self.task_scoring_weights[5]

        if self.debug:
            print("Round-5 Score ", score_5)
            print("Round-5 Weighted Score ", score_5_weighted)

        weighted_score = score_1_weighted + score_2_weighted + \
            score_3_weighted + score_4_weighted + score_5_weighted

        if self.debug:
            print("Weighted Score : ", weighted_score)

        _score_object = {}
        _score_object["score"] = weighted_score * 100
        _score_object["score_secondary"] = score_1 * 100
        _score_object["meta"] = {
            "Task-1-Score": score_1 * 100,
            "Task-2-Score": score_2 * 100,
            "Task-3-Score": score_3 * 100,
            "Task-4-Score": score_4 * 100,
            "Task-5-Score": score_5 * 100,
        }

        return _score_object


if __name__ == "__main__":
    evaluator = NovartisHackathonEvaluator("data/ground_truth", debug=False)
    score_object = evaluator.evaluate("data/submission")
    print(score_object)
