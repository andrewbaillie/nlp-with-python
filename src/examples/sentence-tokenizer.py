from nltk.tokenize import sent_tokenize
from data.bbc_values import text

# Sentence Tokenizer
tokenized_text = sent_tokenize(text)
print(tokenized_text)
