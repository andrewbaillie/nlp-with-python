from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from data.bbc_values import text

stop_words = set(stopwords.words("english"))
lemmatized_words=[]
filtered_sent=[]

# Word Tokenizer
tokenized_word = word_tokenize(text)

for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)

print("Filtered Words:", filtered_sent)
print("")
print("POS Tagged Words:", pos_tag(filtered_sent))
