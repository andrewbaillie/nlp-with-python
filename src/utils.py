from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist

stop_words = set(stopwords.words("english"))
lem = WordNetLemmatizer()


# STRIP PUNCTUATION
# ------------------------------------------------
def strip_punctuation(text):
  translator = str.maketrans('', '', punctuation)
  result = text.translate(translator)
  return result


# STRIP STOPWORDS
# ------------------------------------------------
def strip_stopwords(text):
  filtered = []
  words = word_tokenize(text)
  for w in words:
      if w not in stop_words:
          filtered.append(w)
  return " ".join(filtered)



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


def pos_and_lemmanize(text):
  pos_tagged = pos_tag(word_tokenize(text))
  lemmatized_words = lemmatize_list(pos_tagged)
  return ' '.join(lemmatized_words)

# REMOVE COMMON WORDS
# ------------------------------------------------
def remove_common_words(data):
  words = ' '.join(data)
  words_list = word_tokenize(words)
  fdist = FreqDist(words_list)
  common_words = fdist.most_common(10)

  stripped = []
  for item in data:
    for co in common_words:
      item = item.replace(' ' + co[0] + ' ' , ' ')

    stripped.append(item)
    
  return stripped
  

def remove_short_tweets(tweet, min_length):
  words = word_tokenize(tweet)

  if (len(words) >= min_length):
    return False

  return True
