import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error


def evaluate(prediction, ground_truth):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    pred = np.round(prediction)
    try:
        performance['acc'] = accuracy_score(ground_truth, pred)
    except Exception:
        np.savetxt('prediction', pred, delimiter=',')
        exit(0)
    return performance


def profit_ratio(prediction,ground_truth):
    ground_truth = [abs(i) for i in ground_truth]
    prediction = [int(i) for i in prediction]
    profit = sum([a*b for a,b in zip(ground_truth,prediction)])
    prediction = [1-i for i in prediction]
    loss = sum([a*b for a,b in zip(ground_truth,prediction)])

    return profit/loss