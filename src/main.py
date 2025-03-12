import os
import torch
import numpy as np
import pandas as pd
from data_utils import preprocess_data  # 直接导入，不加 src
from model import get_net, train, get_k_fold_data, log_rmse

def main():
    train_features, train_labels, test_features, test_data = preprocess_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    k = 5
    best_rmse = float('inf')
    best_net = None
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_features, train_labels)
        net = get_net(train_features.shape[1]).to(device)
        rmse = train(net, X_train, y_train)
        if rmse < best_rmse:
            best_rmse = rmse
            best_net = net
            os.makedirs('models', exist_ok=True)
            torch.save(best_net.state_dict(), 'models/best_model.pth')
    with torch.no_grad():
        test_features = test_features.to(device)
        preds = best_net(test_features)
        preds = torch.clamp(preds, 1, float('inf'))
    test_data['SalePrice'] = np.exp(preds.cpu().numpy())
    submission = test_data[['Id', 'SalePrice']]
    os.makedirs('outputs', exist_ok=True)
    submission.to_csv('outputs/submission.csv', index=False)
    with open('outputs/submission_description.txt', 'w') as f:
        f.write(
            "Prediction based on a linear regression model adapted from 'Dive into Deep Learning' (PyTorch edition) by Aston Zhang et al., with personal modifications (e.g., batch_size=128).")
    print(f"Submission file saved to outputs/submission.csv, RMSE: {best_rmse}")
    print("Submission description saved to outputs/submission_description.txt, please include it when submitting to Kaggle.")

if __name__ == "__main__":
    main()