
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.utils import shuffle

class DatasetPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def prepare_datasets(self):
        data_with_manual_review = self.data[self.data['manual_review_case'].notna()]
        data_without_manual_review = self.data[self.data['manual_review_case'].isna()]
        print(f'the length of manual values: {len(data_with_manual_review)}, and the length of non-manual values: {len(data_without_manual_review)}')

        X1 = data_without_manual_review[['recent_cdr', 'recent_PRS_noPOAG']]
        X1_renamed = X1.rename(columns={'recent_cdr': 'cdr', 'recent_PRS_noPOAG': 'PRS_noPOAG'})
        y1 = data_without_manual_review['icd_case'].astype(float)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X1_renamed, y1, test_size=0.2, random_state=42)
        print(f'The shape of train dataset is {self.X_train.shape}, y_train: {self.y_train.shape}, X_val: {self.X_val.shape}, y_val: {self.y_val.shape}')

    def print_summary(self):
        if self.X_train is not None:
            print(f'Training set X shape: {self.X_train.shape}, Training set y shape: {self.y_train.shape}')
            print(f'Validation set X shape: {self.X_val.shape}, Validation set y shape: {self.y_val.shape}')
        else:
            print("Data has not been prepared. Please call the prepare_datasets method first.")
