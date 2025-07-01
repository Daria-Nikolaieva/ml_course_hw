import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_dataset(path):
    """
    Loads the dataset from a CSV file.

    Parameters:
    path (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)

def split_dataset(df, target_col, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and validation sets.

    Parameters:
    df (DataFrame): Original dataset.
    target_col (str): Name of the target column.
    test_size (float): Proportion of the validation set.
    random_state (int): Random seed for reproducibility.

    Returns:
    Tuple: train_df, val_df
    """
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])

def select_features(df, input_cols, target_col):
    """
    Splits the DataFrame into input features and target.

    Parameters:
    df (DataFrame): Input dataset.
    input_cols (list): List of feature column names.
    target_col (str): Name of the target column.

    Returns:
    Tuple: inputs (X), targets (y)
    """
    return df[input_cols].copy(), df[target_col].copy()

def get_column_types(df):
    """
    Identifies numeric and categorical columns in the dataset.

    Parameters:
    df (DataFrame): The input features.

    Returns:
    Tuple: numeric_cols, categorical_cols
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def scale_numeric_columns(train_inputs, val_inputs, numeric_cols):
    """
    Applies MinMax scaling to numeric columns.

    Parameters:
    train_inputs (DataFrame): Training features.
    val_inputs (DataFrame): Validation features.
    numeric_cols (list): List of numeric column names.

    Returns:
    Tuple: Scaled train_inputs and val_inputs
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs

def encode_categorical_columns(train_inputs, val_inputs, categorical_cols):
    """
    Applies One-Hot Encoding to categorical columns.

    Parameters:
    train_inputs (DataFrame): Training features.
    val_inputs (DataFrame): Validation features.
    categorical_cols (list): List of categorical column names.

    Returns:
    Tuple: train_inputs, val_inputs, encoded column names
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    return train_inputs, val_inputs

def preprocess_bank_data(path='train.csv'):
    """
    Performs full preprocessing pipeline on the bank dataset.

    Parameters:
    path (str): Path to the dataset.

    Returns:
    Tuple: train_inputs, val_inputs, train_targets, val_targets, numeric_cols, encoded_cols
    """
    df = load_dataset(path)
    train_df, val_df = split_dataset(df, target_col='Exited')
    
    input_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                  'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    target_col = 'Exited'

    train_inputs, train_targets = select_features(train_df, input_cols, target_col)
    val_inputs, val_targets = select_features(val_df, input_cols, target_col)

    numeric_cols, categorical_cols = get_column_types(train_inputs)

    train_inputs, val_inputs = scale_numeric_columns(train_inputs, val_inputs, numeric_cols)
    train_inputs, val_inputs = encode_categorical_columns(train_inputs, val_inputs, categorical_cols)

    return {
        'train_inputs': train_inputs,
        'val_inputs': val_inputs,
        'train_targets': train_targets,
        'val_targets': val_targets,
    }