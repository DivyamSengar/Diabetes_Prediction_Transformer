import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
class DiabetesDataset(Dataset):
    def __init__(self, data_path, split='train', test_size=0.15, val_size=0.15,
                 random_state=42, augment=False):
        self.data = self._load_and_preprocess(data_path)
        self.categorical_cols = ['smoking_history']
        self.numerical_cols = ['gender','age', 'hypertension', 'heart_disease', 
                              'bmi', 'HbA1c_level', 'blood_glucose_level']
        self.target_col = 'diabetes'
        self.augment = augment
        
        # Split data with stratification
        train_val, test_data = train_test_split(
            self.data, test_size=test_size, random_state=random_state, 
            stratify=self.data[self.target_col]
        )
        
        train_data, val_data = train_test_split(
            train_val, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=train_val[self.target_col]
        )
        
        if split == 'train':
            self.df = train_data
            if self.augment:
                self._apply_smote()
        elif split == 'val':
            self.df = val_data
        else:
            self.df = test_data
        
        # Preprocess data
        self._fit_preprocessors()
        self._preprocess_data()
        # print(self.df.isna().sum())


    def _load_and_preprocess(self, data_path):
        df = pd.read_csv(data_path)
        # print(df.isna().sum())
        # Explicit gender encoding
        # print('unique print: ', df['gender'].unique())
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'female': 0, 'male': 1, 'Other': 0.5, 'other': 0.5})
        
        # Clean smoking history
        df['smoking_history'] = df['smoking_history'].replace({
            'No Info': 'unknown',
            'not current': 'former'  # Merge similar categories
        })
        # # Check for unexpected gender values
        # unique_genders = df['gender'].unique()
        # expected_genders = [0, 1, None]  # Include None for potential NaNs after mapping
        # unexpected_genders = [g for g in unique_genders if g not in expected_genders]
        
        # if unexpected_genders:
        #     print(f"Unexpected gender values found: {unexpected_genders}")
            
        #     original_unique_genders = df['gender'].map({0:'female', 1:'male', None:None}).unique()
        #     original_unexpected_genders = [g for g in original_unique_genders if g not in ['female','male']]
        #     if original_unexpected_genders:
        #         print(f"Original unexpected gender values found before mapping: {original_unexpected_genders}")
        return df

    def _apply_smote(self):
        """Apply SMOTE only to training data"""
        # Separate features and target
        X = self.df[self.categorical_cols + self.numerical_cols]
        y = self.df[self.target_col]
        
        # Apply SMOTE
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        
        self.df = pd.DataFrame(X_res, columns=X.columns)
        self.df[self.target_col] = y_res

    def _fit_preprocessors(self):
        # Fit label encoders and scalers on training data
        self.encoders = {
            'smoking_history': LabelEncoder().fit(self.data['smoking_history'])
        }
        # for col in self.numerical_cols:
        #     lower = self.data[col].quantile(0.01)
        #     upper = self.data[col].quantile(0.99)
        #     self.data[col] = self.data[col].clip(lower, upper)
        self.scalers = {col: StandardScaler().fit(self.data[[col]]) 
                      for col in self.numerical_cols}
        # self.scalers = {col: RobustScaler().fit(self.data[[col]]) 
        #               for col in self.numerical_cols}

    def _preprocess_data(self):
        # Transform data
        # self.df['smoking_history'] = self.encoders['smoking_history'].transform(
        #     self.df['smoking_history']
        # )
        # New one-hot encoding for smoking_history:
        one_hot = pd.get_dummies(self.df['smoking_history'], prefix='smoking_history')
        self.df = pd.concat([self.df, one_hot], axis=1)
        self.one_hot_columns = list(one_hot.columns)
        # Optionally drop the original column:
        self.df.drop(columns=['smoking_history'], inplace=True)
        
        for col in self.numerical_cols:
            self.df[col] = self.scalers[col].transform(self.df[[col]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # categorical = torch.tensor([
        #     # row['gender'],  # Already encoded as 0/1
        #     row['smoking_history']
        # ], dtype=torch.long)
        categorical = torch.tensor(row[self.one_hot_columns].values.astype(np.float32))
        numerical = torch.tensor(row[self.numerical_cols].values.astype(np.float32))
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        
        return {
            'categorical': categorical,
            'numerical': numerical,
            'target': target
        }

def get_dataloaders(data_path, batch_size=32, augment=False):
    train_dataset = DiabetesDataset(data_path, split='train', augment=augment)
    val_dataset = DiabetesDataset(data_path, split='val')
    test_dataset = DiabetesDataset(data_path, split='test')

    # Handle class imbalance with weighted sampling
    class_counts = train_dataset.df['diabetes'].value_counts().to_list()
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.df['diabetes'].values]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # return train_loader, val_loader, test_loader, train_dataset.encoders
    return train_loader, val_loader, test_loader, train_dataset.one_hot_columns