
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from easydict import EasyDict as edict


class OverallEvaluator:
        
    def __init__(self,File_Dict,Weight_Dict,Round):
    
        """
        Master_Step2 is the ground truth for primary keys - CSV
        Participant_Step2 is the CSV file that participant submits
        Comparison_Result_Step1 is the output of the comparison file, this file will have rows populated if there is a
        difference between the files
        Weight2 is the weightage for 2nd step

        Cluster_Data is the CSV dataset shared by the participant that has the predicted value in the "Market_Cluster"
        column for each "Hospital_ID" along with all features used for clustering
        Weight3 is the weight for the 3rd step

        Master_Step4 is the CSV that contains the actual sales for 12 months of the test list of hospitals
        Predict is the CSV shared by the participant containing the predicted value of sales for the provided list of hospitals
        Weight4 is the weight for hte 4th step

        Master_Step5 is the CSV that has the actual sales of the all the hospitals for 12 months
        Optimize is the CSV that contains the optimized Sales for 12 months for all hospitals

        round talks about the different rounds we have
        round = 2 for identifying the primary key
        round = 3 for calculating the silhouette score
        round = 4 for calculating the MAPE
        round = 5 for calculating the score for optimization

        """
    
        if Round == 2:
            self.Master_Step2 = pd.read_csv(File_Dict.Master_Step2,'r')
            self.Participant_Step2 = pd.read_csv(File_Dict.Participant_Step2,'r')
            self.Weight2 = Weight_Dict.Weight2
        elif Round == 3:
            self.Cluster_Data = pd.read_csv(File_Dict.Cluster_Data)
            self.Weight3 = Weight_Dict.Weight3
        elif Round == 4:
            self.Master_Step4 = pd.read_csv(File_Dict.Master_Step4)
            self.Predict = pd.read_csv(File_Dict.Predict)
            self.Weight4 = Weight_Dict.Weight4
        elif Round == 5:
            self.Master_Step5 = pd.read_csv(File_Dict.Master_Step5)
            self.Optimize = pd.read_csv(File_Dict.Optimize)
            self.Weight5 = Weight_Dict.Weight5
        else:
            print('Please validate the input to round')
      
                                       
    def _PrimaryKey(self, client_payload):
        """
        This function will validate the primary key for every file shared
        `client_payload` will be a dict with (atleast) the following keys :
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
        
        Evaluation_Score = 0
                
        with open(File_Dict.Master_Step2, 'r') as df_master, open(File_Dict.Participant_Step2, 'r') as df:
            fileone = df_master.readlines()
            filetwo = df.readlines()
        Line_Count = 0
        with open(File_Dict.Comparison_Result_Step2,'w') as outFile:
            for line in filetwo:
                if line not in fileone:
                    outFile.write(line)
                    Line_Count = Line_Count + 1
            if Line_Count > 0:
                print("The participant needs to rework on the solution.Mismatch in" ,line_count, "lines")
            else:
                print("The participant's file matches completely with the master file")

        # Step Score
        Evaluation_Score = (Total_File_Count.Count - Line_Count)* Weight_Dict.Weight2
        print(Evaluation_Score)
        return Evaluation_Score
        
    def _SilhouetteScore(self,client_payload, Evaluation_Score):
        """
         This function evaluate teh Silhouette score for the clustering dataset provided by the participant
         `client_payload` will be a dict with (atleast) the following keys :
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        
                                       
        #importing the clustered dataset
        df_cluster = pd.read_csv(File_Dict.Cluster_Data)
        
        #Extracting the number of clusters
        n_cluster = df_cluster['Market_Cluster'].max()
        labels = df_cluster['Market_Cluster'].values
                                       
        #Dropping the hospital_id column
        df_cluster = df_cluster.drop(['Hospital_ID','Market_Cluster'], axis=1)

        #Standardizing the dataframe
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(df_cluster)
        
        #Silhouette Score
        cluster_score = silhouette_score(X,labels, metric ='euclidean')
        cluster_score = (cluster_score + 1)
        print ("The the participant's cluster score is : ", cluster_score)

        #Step score
        Silhouette_Score = cluster_score * Weight_Dict.Weight3 / np.log(n_cluster + (np.exp(2) -1))
        print(Silhouette_Score)
        Evaluation_Score += Silhouette_Score
        return Evaluation_Score
        
        
    def _Error(self,client_payload,Evaluation_Score):                                   
        """
        This function will calculate the MAPE for the list of hospitals provided
        
        """
        
        #importing participant dataframe
        predict = pd.read_csv(File_Dict.Predict)
        actual = pd.read_csv(File_Dict.Master_Step4)

        # merging dataframes
        df_mape = pd.merge(predict, actual, on=['Hospital_ID','Month'], how='outer')
        df_mape = df_mape.fillna(0)

        #MAPE
        df_mape['ABS'] = (df_mape['Actual_Sales'] - df_mape['Predict_Sales']).abs()/df_mape['Actual_Sales']
        MAPE = df_mape['ABS'].mean()

        MAPE_Score = (1-MAPE) * Weight_Dict.Weight4
        print(MAPE_Score)
        Evaluation_Score += MAPE_Score
        return MAPE_Score, Evaluation_Score
        
                                       
    
    def _Optimize(self,client_payload,MAPE_Score,Evaluation_Score):
        
        """
        This funciton generates the score for the overall optimized value
        """
        
        # Read optimized value from each participant
        Optimize = pd.read_csv(File_Dict.Optimize, header = None).dropna()
        Optimize_v = Optimize.values
        Optimize_Scalar = np.asscalar(Optimize_v)

        #actual 12 months sales
        Actual = pd.read_csv(File_Dict.Master_Step5,header = None).dropna()
        Actual_v = Actual.values
        Actual_Scalar = np.asscalar(Actual_v)
        
        # Error in Sales
        Error_Score = Optimize_Scalar*(1-MAPE_Score)*Weight_Dict.Weight5 / Actual_Scalar                         
        print(Error_Score)
        Evaluation_Score += Error_Score
        return Evaluation_Score
                 
                                       
if __name__ == "__main__":
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    File_Dict = edict()
    File_Dict['Master_Step2'] = 'Master_Step2.csv'
    File_Dict['Participant_Step2'] = 'Participant_Step2.csv'
    File_Dict['Comparison_Result_Step2'] = 'Comparison_Result_Step2.csv'
    File_Dict['Cluster_Data'] = 'Cluster_Data.csv'
    File_Dict['Master_Step4'] = 'Master_Step4.csv'
    File_Dict['Predict'] = 'Predict.csv'
    File_Dict['Master_Step5'] = 'Master_Step5.csv'
    File_Dict['Optimize'] = 'Optimize.csv'
    
    Weight_Dict = edict()
    Weight_Dict['Weight2'] = 0.1
    Weight_Dict['Weight3'] = 0.3
    Weight_Dict['Weight4'] = 0.35
    Weight_Dict['Weight5'] = 0.15
    
    Total_File_Count = edict()                                   
    Total_File_Count['Count'] = 21                                   

    _client_payload = {}
    _client_payload["aicrowd_submission_id"] = 1123
    _client_payload["aicrowd_participant_id"] = 1234

######################################################################################################################
    
    # Please use this line to evaluate the particular round
    Round = 2

######################################################################################################################
                                       
    # Instantiate an evaluator
    aicrowd_evaluator = OverallEvaluator(File_Dict,Weight_Dict,Round = Round)
    
    # Evaluate
    Evaluation_Score = aicrowd_evaluator._PrimaryKey(_client_payload)
    print("Round:",Round, " ---- Evaluation Score:", Evaluation_Score)
    Round +=1 

    Evaluation_Score = aicrowd_evaluator._SilhouetteScore(_client_payload,Evaluation_Score)
    print("Round:",Round, " ---- Evaluation Score:", Evaluation_Score)                                   
    Round +=1 

    MAPE_Score, Evaluation_Score = aicrowd_evaluator._Error(_client_payload,Evaluation_Score)
    print("Round:",Round, " ---- Evaluation Score:", Evaluation_Score)                                   
    Round +=1 
    
    Evaluation_Score = aicrowd_evaluator._Optimize(_client_payload,MAPE_Score,Evaluation_Score)
    print("Round:",Round, " ---- Evaluation Score:", Evaluation_Score)                                   

