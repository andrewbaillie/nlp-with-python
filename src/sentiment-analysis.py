import pandas as pd
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn import metrics
from utils import strip_punctuation, strip_stopwords, remove_common_words, pos_and_lemmanize, remove_short_tweets

data = pd.read_csv('airline-sentiment.csv')

# Remove Neutral sentiment items from dataset
data = data[(data['airline_sentiment'] != 'neutral')]

# Print the head of the dataset
#print(data.head())

# Get info on the dataset
#data.info()

# Get info on the sentiments
#print(data.airline_sentiment.value_counts())

# Sentiment_count = data.groupby('airline_sentiment').count()
# plt.bar(Sentiment_count.index.values, Sentiment_count['text'])
# plt.xlabel('Tweet Sentiments')
# plt.ylabel('Number of Tweets')
# plt.show()

# ----------------------------------------------------------------
# Prediction Data
# ----------------------------------------------------------------
predictions = [
  "@VirginAmerica plus you've added commercials to the experience... tacky.",
  "@VirginAmerica So excited for my first cross country flight LAX to MCO I've heard nothing but great things about Virgin America. #29DaysToGo",
  "@USAirways if it was so important, why did I wait on hold and then get hung up on by your computer?  #disappointed",
  "@AmericanAir no. Booked seat in Dallas, live in Dallas. Real nice that your gate agent had exit row available told me they weren't available"
]

# ----------------------------------------------------------------
# Vectorizing Data: Bag-Of-Words
# ----------------------------------------------------------------
input_data_set = data.text
input_sentiments = data.airline_sentiment

new_data_set = []

for tweet in input_data_set:
  tweet = tweet.lower()
  tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)
  tweet = strip_punctuation(tweet)
  # tweet = strip_stopwords(tweet)
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
# print(metrics.classification_report(test_labels, predictions_mnb))

bnb = BernoulliNB()
model_bnb = bnb.fit(training_data, training_labels)
predictions_bnb = model_bnb.predict(test_data)
print("BernoulliNB Accuracy:", metrics.accuracy_score(test_labels, predictions_bnb))
# print(metrics.classification_report(test_labels, predictions_bnb))

cnb = ComplementNB()
model_cnb = cnb.fit(training_data, training_labels)
predictions_cnb = model_cnb.predict(test_data)
print("ComplementNB Accuracy:", metrics.accuracy_score(test_labels, predictions_cnb))
# print(metrics.classification_report(test_labels, predictions_cnb))
