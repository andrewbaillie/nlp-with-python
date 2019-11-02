from examples.data.man_united import text
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import pos_tag
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from string import punctuation

stop_words = set(stopwords.words("english"))
lem = WordNetLemmatizer()

totals = {
  'adjectives': 0,
  'nouns': 0,
  'adverbs': 0,
  'verbs': 0
}

# Function: Strip Punctuation
# Description: Strips all punctuation from a given string
def strip_punctuation(text):
  print(punctuation)
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

def penn_to_wn(tag, totals):
    if is_adjective(tag):
        totals['adjectives'] += 1
        return wn.ADJ
    elif is_noun(tag):
        totals['nouns'] += 1
        return wn.NOUN
    elif is_adverb(tag):
        totals['adverbs'] += 1
        return wn.ADV
    elif is_verb(tag):
        totals['verbs'] += 1
        return wn.VERB
    return None

# Function: Lemmatize List
# Description: Lemmatizes words in a list
def lemmatize_list(words, totals):
  lemmatized_words = []
  for w in words:
    tag = penn_to_wn(w[1], totals)
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

lemmatized_words = lemmatize_list(pos_tagged, totals)
print("\nLemmatized Words: ", lemmatized_words)

fdist = FreqDist(lemmatized_words)
common_words = fdist.most_common(20)

print("\nMost Common Words: ", common_words)

print("\nTotal Adjectives: ", totals['adjectives'])
print("\nTotal nouns: ", totals['nouns'])
print("\nTotal adverbs: ", totals['adverbs'])
print("\nTotal verbs: ", totals['verbs'])

wordcloud = WordCloud(width=2000, 
                      height=2000,
                      max_words=100,
                      mode='RGBA',
                      background_color=None
                      ).generate_from_frequencies(fdist)

wordcloud.to_file("../plots/united-wordcloud.png")

# Uncomment to show graph
# fdist.plot(20,cumulative=False)
# plt.show()
