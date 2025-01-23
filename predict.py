import argparse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Parse input arguments
parser = argparse.ArgumentParser(description="Fake News Prediction")
parser.add_argument('--input', type=str, required=True, help="News article text to classify")
args = parser.parse_args()

# Load the trained model and vectorizer
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Vectorize the input text
input_text = [args.input]
input_tfidf = vectorizer.transform(input_text)

# Make prediction
prediction = model.predict(input_tfidf)
prediction_label = 'Fake' if prediction[0] == 0 else 'Real'

# Print the prediction result
print(f"The article is classified as: {prediction_label}")
