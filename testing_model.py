import joblib

# Load the trained model and vectorizer
pac = joblib.load("model.pkl")
tfidf_vectorizer = joblib.load("vectorizer.pkl")

# Take a custom news headline as input
user_input = input("Enter a news headline to check if it's Fake or Real: ")

# Transform input text using the loaded TF-IDF vectorizer
user_tfidf = tfidf_vectorizer.transform([user_input])

# Predict using the trained model
prediction = pac.predict(user_tfidf)

# Display the result
if prediction[0] == 'FAKE':
    print("❌ The news headline is FAKE!")
else:
    print("✅ The news headline is REAL!")
print(prediction)
print(tfidf_vectorizer.transform([user_input]).toarray())
