import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, roc_curve, confusion_matrix, precision_score

def calculate_performance(model, data, target, name):
    predictions = model.predict(data)

    f1 = f1_score(target, predictions)
    accuracy = accuracy_score(target, predictions)
    precision = precision_score(target, predictions)
    model_name = str(name)

    score = pd.DataFrame()

    score['Model'] = pd.Series(model_name)
    score['F1'] = pd.Series(f1)
    score['Accuracy'] = pd.Series(accuracy)
    score['Precision'] = pd.Series(precision)

    return score

def max_seq_length(sequence):
    length = []
    for i in range(0, len(sequence)):
        length.append(len(sequence[i]))
    return max(length)