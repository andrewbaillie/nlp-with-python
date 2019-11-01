from data.bbc_values import text
from string import punctuation

translator = str.maketrans('', '', punctuation)

output = text.translate(translator)
print(output)
