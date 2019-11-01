from examples.data.bbc_values import text
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import pos_tag
from matplotlib import pyplot as plt
from string import punctuation

stop_words = set(stopwords.words("english"))
lem = WordNetLemmatizer()

# Function: Strip Punctuation
# Description: Strips all punctuation from a given string
def strip_punctuation(text):
  translator = str.maketrans('', '', punctuation)
  result = text.translate(translator)
  return result

# Function: Strip Stopwords
# Description: Strips all stopwords from a given List of words
def strip_stopwords(words):
  filtered = []
  for w in words:
      if w not in stop_words:
          filtered.append(w)
  return filtered


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

# Function: Lemmatize List
# Description: Lemmatizes words in a list
def lemmatize_list(words):
  lemmatized_words = []
  for w in words:
    tag = penn_to_wn(w[1]);
    if (tag != None):
      lemmatized_words.append(lem.lemmatize(w[0], tag))
    else:
      lemmatized_words.append(lem.lemmatize(w[0]))
  return lemmatized_words

print("\nOriginal String: ", text)

stripped = strip_punctuation(text)
print("\nStripped Punctuation: ", stripped)

tokenized_words = word_tokenize(stripped)
print("\nTokenized Words: ", tokenized_words)

stripped_stopwords = strip_stopwords(tokenized_words)
print("\nStop Words Stripped: ", stripped_stopwords)

pos_tagged = pos_tag(stripped_stopwords)
print("\nPOS Tagged Words:", pos_tagged)

lemmatized_words = lemmatize_list(pos_tagged)
print("\nLemmatized Words: ", lemmatized_words)

fdist = FreqDist(lemmatized_words)

print("\nMost Common Words: ", fdist.most_common(10))

fdist.plot(30,cumulative=False)
plt.show()
