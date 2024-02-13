import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    true_positive = np.size(y_pred[(np.logical_and(y_pred == 1, y_true == 1))])
    true_negative = np.size(y_pred[np.logical_and(y_pred == 0, y_true == 0)])
    false_positive = np.size(y_pred[np.logical_and(y_pred == 1, y_true == 0)])
    false_negative = np.size(y_pred[np.logical_and(y_pred == 0, y_true == 1)])
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * ((precision * recall) / (precision + recall))
    accuracy = (true_positive + true_negative) / np.size(y_pred)
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    true_positive = np.size(y_pred[(np.where(y_pred == y_true))])
    true_negative = np.size(y_pred[np.where(y_pred != y_true)])

    accuracy = (true_positive + true_negative) / np.size(y_pred)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r_sq = 1 - sum((np.square(y_pred - y_true)))/sum((np.square(y_pred - np.mean(y_true))))
    return r_sq


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mean_squared_error = np.mean(np.square(y_pred - y_true))
    return mean_squared_error


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mean_absolute_error = np.mean(np.abs(y_pred - y_true))
    return mean_absolute_error
    
