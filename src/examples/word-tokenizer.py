from nltk.tokenize import word_tokenize
from data.bbc_values import text

# Word Tokenizer
tokenized_word = word_tokenize(text)
print(tokenized_word)
