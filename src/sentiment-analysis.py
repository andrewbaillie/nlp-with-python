import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.tokenize import RegexpTokenizer

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

# Tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# Generate document term matrix by using scikit-learn's CountVectorizer
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['text'])

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['airline_sentiment'], test_size=0.3, random_state=1)

clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
