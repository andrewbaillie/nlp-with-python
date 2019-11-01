from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

text="""Trust is the foundation of the BBC; we are independent, impartial and honest. Audiences are at the heart of everything we do. We take pride in delivering quality and value for money. Creativity is the lifeblood of our organisation. We respect each other and celebrate our diversity. We are one BBC; great things happen when we work together."""

# Word Tokenizer
tokenized_word = word_tokenize(text)
fdist = FreqDist(tokenized_word)

print(fdist)
print(fdist.most_common(10))

fdist.plot(30,cumulative=False)
plt.show()
