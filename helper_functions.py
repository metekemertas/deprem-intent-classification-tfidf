import numpy as np
import pandas as pd

# References: # https://www.kaggle.com/code/kobakhit/eda-and-multi-label-classification-for-arxiv

# Label based accuracy
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list), acc_list


def feature_importance(pipeline):
    '''
    Extract feature importances from pipeline.
    Since I am using CalibratedClassifierCV I will average the coefficients over calibrated classifiers.
    '''
    # average coefficients over all calibrated classifiers
    coef_avg = 0
    classifiers = pipeline[1].estimators_[0].calibrated_classifiers_
    for i in classifiers:
        coef_avg = coef_avg + i.base_estimator.coef_
    coef_avg = (coef_avg/len(classifiers)).tolist()[0]
    # get feature names from tf-idf vectorizer
    features = pipeline[0].get_feature_names()
    # get 10 most important features
    top_f = pd.DataFrame(list(zip(features,coef_avg)), columns = ['token','coef']) \
        .nlargest(10,'coef').to_dict(orient = 'records')
    return top_f


def predict_tags(X, models, labels=None):
    '''
    Predict tags for a given abstract.

    Args:
      - X (list): an iterable with text.
      - labels (pandas.Dataframe): label indicators for an abstract
    '''
    preds = []
    if type(X) is str:  # convert into iterable if string
        X = [X]

    # get prediction from each model
    for c in models.keys():
        preds.append(models[c].predict(X))

    # print original labels if given
    if labels is not None:
        assert len(X) == 1, 'Only one extract at a time.'
        predicted_tags = [k for k, v in zip(list(models.keys()), preds) if v[0] > 0]
        original_tags = list(labels.index[labels.map(lambda x: x > 0)])
        print('Original Tags: {}'.format(str(original_tags)))
        print("Predicted Tags: {}".format(str(predicted_tags)))

    return preds