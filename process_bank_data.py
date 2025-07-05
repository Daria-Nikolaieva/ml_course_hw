import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_and_split_data(filepath, target_col='Exited', test_size=0.2, random_state=42):
    """
    Loads a CSV dataset and splits it into train and validation sets using stratification.
    """
    df = pd.read_csv(filepath)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return train_df, val_df

def get_input_target(df, input_cols, target_col):
    """
    Splits a DataFrame into input features and target values.
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets

def split_columns_by_type(df):
    """
    Separates column names by numeric and categorical types.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def scale_numeric_columns(train_inputs, val_inputs, numeric_cols):
    """
    Scales numeric features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return scaler, train_inputs, val_inputs

def encode_categorical_columns(train_inputs, val_inputs, categorical_cols):
    """
    One-hot encodes categorical features using OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_encoded = encoder.transform(train_inputs[categorical_cols])
    val_encoded = encoder.transform(val_inputs[categorical_cols])
    train_inputs[encoded_cols] = train_encoded
    val_inputs[encoded_cols] = val_encoded
    return encoder, encoded_cols, train_inputs, val_inputs

def preprocess_bank_data(filepath):
    """
    Full preprocessing pipeline for the bank dataset.
    Returns train/val data and fitted transformers.
    """
    input_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                  'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    target_col = 'Exited'
    
    train_df, val_df = load_and_split_data(filepath, target_col)
    train_inputs, train_targets = get_input_target(train_df, input_cols, target_col)
    val_inputs, val_targets = get_input_target(val_df, input_cols, target_col)
    
    numeric_cols, categorical_cols = split_columns_by_type(train_inputs)
    scaler, train_inputs, val_inputs = scale_numeric_columns(train_inputs, val_inputs, numeric_cols)
    encoder, encoded_cols, train_inputs, val_inputs = encode_categorical_columns(train_inputs, val_inputs, categorical_cols)
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]

    return {
        'train_inputs': train_inputs,
        'train_targets': train_targets,
        'X_train': X_train,
        'X_val': X_val,
        'val_inputs': val_inputs,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cols': encoded_cols,
        'scaler': scaler,
        'encoder': encoder
    }

def preprocess_new_data(new_df, input_cols, numeric_cols, categorical_cols, scaler, encoder):
    """
    Applies preprocessing to new (unseen) data using existing fitted scaler and encoder.
    """
    inputs = new_df[input_cols].copy()
    
    # Масштабируем числовые колонки
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
    
    # One-hot энкодинг категориальных признаков
    encoded = encoder.transform(inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=inputs.index)
    
    # Объединяем числовые и закодированные категориальные признаки
    result_inputs = pd.concat([inputs[numeric_cols], encoded_df], axis=1)
    
    return result_inputs