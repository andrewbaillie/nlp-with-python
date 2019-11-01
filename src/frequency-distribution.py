from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from data.bbc_values import text

# Word Tokenizer
tokenized_word = word_tokenize(text)
fdist = FreqDist(tokenized_word)

print(fdist)
print(fdist.most_common(10))

fdist.plot(30,cumulative=False)
plt.show()
