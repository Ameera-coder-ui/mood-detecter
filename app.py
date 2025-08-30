import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

model = LogisticRegression(max_iter=1000, multi_class="multinomial")


data = pd.read_csv("mood_dataset.csv")

data['mood'] = data['mood'].str.lower()

X = data['text']
y = data['mood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

st.title("Mood Detection App 😊")
user_input = st.text_input("Enter your text:")

if st.button("Predict Mood"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]
        st.write(f"Predicted Mood: **{prediction}**")
    else:
        st.write("⚠️ Please enter some text.")

prediction = model.predict([user_input])[0]
st.write("### Mood Prediction: ", prediction)
