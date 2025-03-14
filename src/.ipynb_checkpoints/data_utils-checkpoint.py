import os
import pandas as pd
import numpy as np
import torch

def preprocess_data(data_dir="data"):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if 'SalePrice' not in train_data.columns:
        raise ValueError("Training data must contain 'SalePrice' column")

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    all_features = pd.get_dummies(all_features, dummy_na=True)

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].fillna(0)  # 处理缺失值
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))  # 避免除以 0

    train_features = all_features[:train_data.shape[0]].values.astype(np.float32)
    test_features = all_features[train_data.shape[0]:].values.astype(np.float32)
    train_labels = torch.tensor(train_data['SalePrice'].values, dtype=torch.float32).view(-1, 1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
    train_labels = train_labels.to(device)

    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    pd.DataFrame(train_features.cpu().numpy()).to_csv(os.path.join(data_dir, "processed/train_features.csv"), index=False)
    pd.DataFrame(test_features.cpu().numpy()).to_csv(os.path.join(data_dir, "processed/test_features.csv"), index=False)
    pd.DataFrame(train_labels.cpu().numpy()).to_csv(os.path.join(data_dir, "processed/train_labels.csv"), index=False)

    return train_features, train_labels, test_features, test_data