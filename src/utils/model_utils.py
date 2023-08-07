from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def model_evaluation(model, X, y):
    y_pred_train = model.predict(X)
    confusion_matrix = metrics.confusion_matrix(y_true=y, y_pred=y_pred_train)
    accuracy = metrics.accuracy_score(y_true=y, y_pred=y_pred_train)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        y_true=y,
        y_pred=y_pred_train
    )

    return {'confusion_matrix': confusion_matrix,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fscore': fscore}


def store_results(X_train, y_train, X_test, y_test, model, folds):
    cross_validation = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=folds,
        n_jobs=-1)

    scores = {'train': [model_evaluation(model, X_train, y_train)],
              'test': [model_evaluation(model, X_test, y_test)],
              'cross-validation': cross_validation}

    return scores


def draw_roc(model, X_test, actual):
    probs = model.predict(X_test)
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
