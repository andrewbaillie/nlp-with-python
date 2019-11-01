from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from data.bbc_values import text

lem = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
lemmatized_words=[]
filtered_sent=[]

# Word Tokenizer
tokenized_word = word_tokenize(text)

for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)

for w in filtered_sent:
    lemmatized_words.append(lem.lemmatize(w))

print("Filtered Words:", filtered_sent)
print("")
print("Lemmatized Words:", lemmatized_words)
