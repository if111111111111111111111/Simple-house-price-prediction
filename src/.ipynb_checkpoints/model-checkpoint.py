import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

def get_net(input_dim):
    return nn.Linear(input_dim, 1)

def log_rmse(net, features, labels):
    with torch.no_grad():
        preds = net(features)
        mse = nn.MSELoss()(preds, labels)
    return torch.sqrt(mse).item()

def train(net, train_features, train_labels, num_epochs=100, learning_rate=0.001, weight_decay=0., batch_size=128):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device).float()
    if not isinstance(train_features, torch.Tensor):
        train_features = torch.tensor(train_features, dtype=torch.float32)
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    train_ds = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    log_data = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = nn.MSELoss()(net(X), y)
            l.backward()
            optimizer.step()
        rmse = log_rmse(net, train_features, train_labels)
        log_data.append([epoch + 1, rmse])
    pd.DataFrame(log_data, columns=['Epoch', 'RMSE']).to_csv('outputs/training_log.csv', index=False)
    return rmse

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx], y[idx]

        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_valid = X_train.to(device), X_valid.to(device)
    y_train, y_valid = y_train.to(device), y_valid.to(device)

    return X_train, y_train, X_valid, y_valid