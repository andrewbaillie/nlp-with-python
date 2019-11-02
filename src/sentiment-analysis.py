import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn import metrics

data = pd.read_csv('airline-sentiment.csv')

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
  "@VirginAmerica plus you've added commercials to the experience... tacky.,,",
  "@VirginAmerica So excited for my first cross country flight LAX to MCO I've heard nothing but great things about Virgin America. #29DaysToGo",
  "@USAirways if it was so important, why did I wait on hold and then get hung up on by your computer?  #disappointed",
  "@AmericanAir no. Booked seat in Dallas, live in Dallas. Real nice that your gate agent had exit row available told me they weren't available"
]

# ----------------------------------------------------------------
# Vectorizing Data: Bag-Of-Words
# ----------------------------------------------------------------

# Generate document term matrix by using scikit-learn's CountVectorizer
cv_bow = CountVectorizer(lowercase=True,
                     stop_words='english',
                     ngram_range = (1,1))

text_cv_bow = cv_bow.fit_transform(data['text'])

# We now have a test set (test_data) that represents 33% of the original dataset.
training_data, test_data, training_labels, test_labels = train_test_split(
    text_cv_bow, 
    data['airline_sentiment'], 
    test_size=0.33, 
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
