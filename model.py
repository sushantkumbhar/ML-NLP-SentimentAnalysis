import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
#import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import re
import string

train_data=pd.read_csv('./train.csv')
print("Train Data-:\n",train_data.head())
test_data=pd.read_csv('./test.csv')
print("Test Data-:\n",test_data.head())

print("Value Counts \n",train_data['Is_Response'].value_counts())
train_data.drop(columns=['User_ID', 'Browser_Used', 'Device_Used'], inplace=True)

def text_clean(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[""''_]', '', text)
    text = re.sub('\n', '', text)
    return text


def decontract_text(text):
    """
    Decontract text
    """
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"won\’t", "will not", text)
    text = re.sub(r"can\’t", "can not", text)
    text = re.sub(r"\'t've", " not have", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'clock", "f the clock", text)
    text = re.sub(r"\'cause", " because", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = re.sub(r"n\’t", " not", text)
    text = re.sub(r"\’re", " are", text)
    text = re.sub(r"\’s", " is", text)
    text = re.sub(r"\’d", " would", text)
    text = re.sub(r"\’ll", " will", text)
    text = re.sub(r"\’t", " not", text)
    text = re.sub(r"\’ve", " have", text)
    text = re.sub(r"\’m", " am", text)

    return text
train_data['cleaned_description'] = train_data['Description'].apply(lambda x: decontract_text(x))
train_data['cleaned_description'] = train_data['cleaned_description'].apply(lambda x: text_clean(x))

print("Cleaned Data:",train_data['cleaned_description'][0])

x, y = train_data['cleaned_description'], train_data['Is_Response']
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.1,
                                                    random_state=42)

print(f'x_train: {len(x_train)}')
print(f'x_test: {len(x_test)}')
print(f'y_train: {len(y_train)}')
print(f'y_test: {len(y_test)}')

tvec = TfidfVectorizer()
clf = LogisticRegression(solver='lbfgs', max_iter=1000)
model = Pipeline([('vectorizer', tvec), ('classifier', clf)]) #First run vectorizer and Logistic classifier
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Accurcy: {accuracy_score(y_pred, y_test)}')
print(f'Precision: {precision_score(y_pred, y_test, average="weighted")}')
print(f'Recall: {recall_score(y_pred, y_test, average="weighted")}')

print("Confusion Matrix:",confusion_matrix(y_test, y_pred))

print("Test Predict:",model.predict(["i m awesome"]))

#Saving model to disk
pickle.dump(model,open('model.pkl','wb'))

model1=pickle.load(open('model.pkl','rb'))
print("Test Predict:",model1.predict(["i m awesome"]))


