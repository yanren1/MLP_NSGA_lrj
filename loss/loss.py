import torch

def weighted_mse(y_pred, y_true, weights):

    # max_vals, _ = torch.max(y_true, dim=0)
    #
    # y_pred,y_true = y_pred/max_vals,y_true/max_vals

    return torch.mean((weights * torch.abs(y_pred - y_true))**2)


def MAPE(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true+1e-6)) * 100