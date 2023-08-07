from sklearn import metrics
from sklearn.model_selection import cross_val_score


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

