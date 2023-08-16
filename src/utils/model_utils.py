import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def model_evaluation(model, X, y):
    y_pred = model.predict(X)
    accuracy = metrics.accuracy_score(y, y_pred)
    cnf_matrix = metrics.confusion_matrix(y, y_pred)

    return accuracy, cnf_matrix


def precision_recall(confusion_matrix):
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    precision = TP / (TP + FN)
    recall = TP / (TP + FP)

    return precision, recall

def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def store_results(name, X_train, y_train, X_test, y_test, model):

    accuracy_train, cm_train = model_evaluation(model, X_train, y_train)
    accuracy_test, cm_test = model_evaluation(model, X_test, y_test)

    precision_train, recall_train = precision_recall(cm_train)
    precision_test, recall_test = precision_recall(cm_test)

    f1_train, f1_test = f1_score(precision_train, recall_train), f1_score(precision_test, recall_test)

    output = {
        'model-name': [name],
        'accuracy-train': [accuracy_train],
        'precision-train': [precision_train],
        'recall-train': [recall_train],
        'f1-train': [f1_train],
        'accuracy_test': [accuracy_test],
        'precision-test': [precision_test],
        'recall-test': [recall_test],
        'f1-test': [f1_test]
    }

    print(f'Training data confusion matrix: \n {cm_train}')
    print(f'Test data confusion matrix: \n {cm_test}')

    return pd.DataFrame(output)


def draw_roc(model, x, actual):
    probs = model.predict(x)
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds
