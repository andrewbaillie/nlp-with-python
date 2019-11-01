from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from data.bbc_values import text

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
stemmed_words=[]
filtered_sent=[]

# Word Tokenizer
tokenized_word = word_tokenize(text)

for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)

for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)
