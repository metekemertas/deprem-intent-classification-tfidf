import json
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_curve, auc, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import helper_functions
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

# References: # https://www.kaggle.com/code/kobakhit/eda-and-multi-label-classification-for-arxiv


# All Labels
etiketler = ["Arama Ekipmani", "Enkaz Kaldirma", "Isinma", "Elektrik Kaynagi", "Giysi", "Yemek", "Su",
             "Barınma", "Saglik", "Lojistik", "Tuvalet", "Alakasiz", "Cenaze"]

# ----------------------------------------------------------------------------------------------------------------------
# Read Data
with open('full-10140-labeled-gigantic-bandicoot.jsonl', 'r') as json_file:
    json_list = list(json_file)

tutarli = {}
for i in range(len(json_list)):
    j_d = json.loads(json_list[i])
    tutarli[i] = j_d

print(tutarli[1])

# ----------------------------------------------------------------------------------------------------------------------
# Label distribution
tutarli_labels = []
for i in list(tutarli.keys()):
    tutarli_labels.append(tutarli[i]['label'])

# list of list to list
tutarli_labels_2 = [item for sub in tutarli_labels for item in sub]

etiket_count = {}
for et in etiketler:
    etiket_count[et] = tutarli_labels_2.count(et)
print(etiket_count)

fig = plt.figure()
plt.bar(etiket_count.keys(), etiket_count.values(), color='g')
plt.xticks(rotation=90)
plt.title('Etiket Dagilimi')
fig.tight_layout()
plt.savefig('etiket_count.png')

# ----------------------------------------------------------------------------------------------------------------------

# Preprocess
df = pd.DataFrame.from_dict(tutarli).T
mask = df.label.apply(lambda x: 'Alakasiz' not in x)
df = df[mask]
df = df.reset_index()
df['index'] = list(range(0, len(df)))
df.drop('index', axis=1, inplace=True)

# get labels
mlb = MultiLabelBinarizer()
mlb_labels = mlb.fit_transform(df.label)

df_2 = pd.concat([df, pd.DataFrame(mlb_labels)], axis=1)
df_2.columns = ['image_url', 'label', 'label_confidence', 'labeler', 'label_creation_time'] + list(mlb.classes_)

categories = df_2.columns[5:]

trainidx, validx = list(IterativeStratification(n_splits=10, order=1).split(X=mlb_labels, y=mlb_labels))[0]
df_train_splitted = df_2.iloc[trainidx].reset_index(drop=True)
df_val_splitted = df_2.iloc[validx].reset_index(drop=True)
print('train data: ', len(df_train_splitted), 'valid data: ', len(df_val_splitted))

X_train = df_train_splitted.image_url
X_test = df_val_splitted.image_url

# ----------------------------------------------------------------------------------------------------------------------
# define the pipeline
# classifier = CalibratedClassifierCV(LinearSVC())
# classifier = CalibratedClassifierCV(DecisionTreeClassifier(max_depth=10))
classifier = CalibratedClassifierCV(xgb.XGBClassifier(scale_pos_weight=50))  # Best
# classifier = CalibratedClassifierCV(xgb.XGBClassifier(scale_pos_weight=75, learning_rate=0.02, gamma=3, max_depth=8))
# classifier = CalibratedClassifierCV(xgb.XGBClassifier(scale_pos_weight=100))
# classifier = CalibratedClassifierCV(xgb.XGBClassifier(scale_pos_weight=25))

# for each category train the model and get accuracy, auc
models = {}
features = {}
preds = {}
for category in categories:
    # give pipelines unique names.
    SVC_pipeline = Pipeline([
        (f'tfidf_{category}', TfidfVectorizer()),
        (f'clf_{category}', OneVsRestClassifier(classifier, n_jobs=1)),
    ])
    print('... Processing {}'.format(category))

    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, df_train_splitted[category])
    models[category] = SVC_pipeline

    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    preds[category] = prediction
    accuracy = accuracy_score(df_val_splitted[category], prediction)

    # compute auc
    probas_ = SVC_pipeline.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(df_val_splitted[category], probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Accuracy : {} . Area under the ROC curve : {}".format(round(accuracy, 4), round(roc_auc, 4)))
    print()


# ----------------------------------------------------------------------------------------------------------------------
# predict tags for 20 image_urls
for i in range(0, 20):
    print(df_val_splitted.image_url.iloc[i])
    helper_functions.predict_tags(df_val_splitted['image_url'].iloc[i], models, labels=df_val_splitted.iloc[i, 5:])
    print()

# get all predictions
y_pred = np.array(helper_functions.predict_tags(df_val_splitted.image_url, models)).T
# get true labels in the same order
y_true = df_val_splitted[list(models.keys())].to_numpy()

# Scores
hamming_loss_v = hamming_loss(y_true, y_pred) # fraction of labels assgined incorrectly. the lower the better
acc_ = accuracy_score(y_true, y_pred)
mean_score, list_score = helper_functions.hamming_score(y_true, y_pred)

# Classification Report
print(classification_report(y_true, y_pred, target_names=list(categories)))

print('done')

"""
xgboost classifier (Alakasiz hariç)
scale_pos_weight=50  -->best

                  precision    recall  f1-score   support

  Arama Ekipmani       0.84      0.59      0.70        64
         Barınma       0.88      0.97      0.92       231
          Cenaze       0.00      0.00      0.00         2
Elektrik Kaynagi       0.86      0.80      0.83        30
  Enkaz Kaldirma       0.92      0.98      0.95       468
           Giysi       0.91      0.73      0.81        71
          Isinma       0.85      0.82      0.83        88
        Lojistik       0.92      0.61      0.73        18
          Saglik       0.82      0.56      0.67        57
              Su       0.86      0.78      0.82        32
         Tuvalet       0.80      0.80      0.80         5
           Yemek       0.94      0.91      0.92       107

       micro avg       0.90      0.88      0.89      1173
       macro avg       0.80      0.71      0.75      1173
    weighted avg       0.90      0.88      0.89      1173
     samples avg       0.91      0.91      0.90      1173
"""
