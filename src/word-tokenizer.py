from nltk.tokenize import word_tokenize

text="""Trust is the foundation of the BBC; we are independent, impartial and honest. Audiences are at the heart of everything we do. We take pride in delivering quality and value for money. Creativity is the lifeblood of our organisation. We respect each other and celebrate our diversity. We are one BBC; great things happen when we work together."""

# Word Tokenizer
tokenized_word = word_tokenize(text)
print(tokenized_word)
