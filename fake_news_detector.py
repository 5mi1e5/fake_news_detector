import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Read the data
df = pd.read_csv('c:\\Users\\Adn\\Desktop\\news2.csv', usecols=['title', 'text', 'subject', 'date', 'labels'])

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['labels'], test_size=0.2, random_state=7)
x_train = x_train.fillna("")  
x_test = x_test.fillna("")  

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

mask = y_train.notna()
x_train = x_train[mask]
y_train = y_train[mask]

# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
# Ensure y_test and y_pred are strings (since labels are often categorical)
y_test = y_test.astype(str)
y_pred = y_pred.astype(str)

# Calculate accuracy
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

import joblib

#Save the trained model and vectorizer
joblib.dump(pac, "model.pkl")  
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved!")
