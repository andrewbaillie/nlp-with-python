import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn import metrics
from utils import strip_punctuation, strip_stopwords, remove_common_words, pos_and_lemmanize, remove_short_tweets


ENABLE_NEUTRAL_SENTIMENTS = True
ENABLE_DATA_SANITIZATION = True
ENABLE_CLASSIFICATION_REPORT = False
ENABLE_CUSTOM_PREDICTIONS = True

data = pd.read_csv('airline-sentiment.csv')

# Remove Neutral sentiment items from dataset
if (ENABLE_NEUTRAL_SENTIMENTS):
  print("\n*****\nRemoving Neutral Sentiments from Dataset\n*****\n")
  data = data[(data['airline_sentiment'] != 'neutral')]

# Print the head of the dataset
#print(data.head())

# Get info on the dataset
#data.info()

# Get info on the sentiments
# print(data.airline_sentiment.value_counts())

# Sentiment_count = data.groupby('airline_sentiment').count()
# plt.bar(Sentiment_count.index.values, Sentiment_count['text'])
# plt.xlabel('Tweet Sentiments')
# plt.ylabel('Number of Tweets')
# plt.show()

# ----------------------------------------------------------------
# Prediction Data
# ----------------------------------------------------------------
predictions = [
  "@BritishAirways y u lyin?",
  "@AmericanAir hey I’m 24 hours before a flight and got an email to check in and I can’t check in on the site. The button to check in is not there.",
  "@United_Airline @united thank you for your help today! You’ve saved my day!",
  "The customer care and response rate of @flyethiopian is really wanting and totally frustrating!!!",
  "@TK_HelpDesk Great, thank you looking forward to visiting Turkey for the first time"
]


# ----------------------------------------------------------------
# Vectorizing Data: Bag-Of-Words
# ----------------------------------------------------------------
input_data_set = data.text
input_sentiments = data.airline_sentiment

if (ENABLE_DATA_SANITIZATION):
  new_data_set = []
  print("\n*****\nEnabling Data Sanitization\n*****\n")

  for tweet in input_data_set:
    tweet = tweet.lower()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
    tweet = strip_punctuation(tweet)
    tweet = pos_and_lemmanize(tweet)
    new_data_set.append(tweet)

  # Remove common words
  new_data_set = remove_common_words(new_data_set)

  # Remove short tweets
  i = 0
  while i < len(new_data_set):
    if (remove_short_tweets(new_data_set[i], 1)):
      del new_data_set[i]
      del input_sentiments[i]
    i += 1

else: 
  new_data_set = input_data_set

# Generate document term matrix by using scikit-learn's CountVectorizer
cv_bow = CountVectorizer(lowercase=True, ngram_range = (1,1))
text_cv_bow = cv_bow.fit_transform(new_data_set)

# We now have a test set (test_data) that represents 20% of the original dataset.
training_data, test_data, training_labels, test_labels = train_test_split(
    text_cv_bow, 
    input_sentiments, 
    test_size=0.2, 
    random_state=1)

mnb = MultinomialNB()
model = mnb.fit(training_data, training_labels)
predictions_mnb = model.predict(test_data)
print("MultinomialNB Accuracy:", metrics.accuracy_score(test_labels, predictions_mnb))


bnb = BernoulliNB()
model_bnb = bnb.fit(training_data, training_labels)
predictions_bnb = model_bnb.predict(test_data)
print("BernoulliNB Accuracy:", metrics.accuracy_score(test_labels, predictions_bnb))


cnb = ComplementNB()
model_cnb = cnb.fit(training_data, training_labels)
predictions_cnb = model_cnb.predict(test_data)
print("ComplementNB Accuracy:", metrics.accuracy_score(test_labels, predictions_cnb))


if (ENABLE_CUSTOM_PREDICTIONS):

  print("\n\n******************************\n")

  # Generate vectors for custom predictions
  text_custom = cv_bow.transform(predictions)

  multinomial_predictions = model.predict(text_custom)
  bernoulli_predictions = model_bnb.predict(text_custom)
  complement_predictions = model_cnb.predict(text_custom)

  a = 0
  while a < len(predictions):
    print("\n", predictions[a], "\n", multinomial_predictions[a], " - ", bernoulli_predictions[a], " - ", complement_predictions[a])
    a += 1



if (ENABLE_CLASSIFICATION_REPORT):
  print("\n*****************************************************\n")
  print("\nClassification Report for MultinomialNB:\n")
  print(metrics.classification_report(test_labels, predictions_mnb))

  print("\n\nClassification Report for BernoulliNB:\n")
  print(metrics.classification_report(test_labels, predictions_bnb))

  print("\n\nClassification Report for ComplementNB:\n")
  print(metrics.classification_report(test_labels, predictions_cnb))
