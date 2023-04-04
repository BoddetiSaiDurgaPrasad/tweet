import tweepy
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Set up Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'
# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# Search for tweets using a hashtag or keyword
query = 'your_query'
tweets = api.search_tweets(q=query, count=100)
# Preprocess and clean the text data
stop_words = set(stopwords.words('english'))
processed_tweets = []
for tweet in tweets:
    text = tweet.text.lower()
    text = re.sub(r"http\S+", "", text) # Remove URLs
    text = re.sub('[^a-zA-Z0-9\s]', '', text) # Remove special characters
    text = re.sub('\n', '', text) # Remove new line characters
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Remove stop words
    processed_tweets.append(text)
# Convert the text data to a bag-of-words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_tweets)
# Load the sentiment labels for the training data
sentiment_data = pd.read_csv(r'"C:\Users\Sai Durga Prasad\Downloads\archive\twitter_training.csv"')
y = sentiment_data['Sentiment']
# Train a Naive Bayes classifier on the labeled data
clf = MultinomialNB()
clf.fit(X, y)
# Classify the tweets using the trained classifier
predictions = clf.predict(X)
# Print the accuracy of the classifier on the labeled data
accuracy = accuracy_score(y, predictions)
print("Accuracy: ", accuracy)
