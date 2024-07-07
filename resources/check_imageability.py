
# %%
from utils import *
from functions import *

# %%

# Script for making the lancaster sensorimotor norms into a dictionary (lemmatizing etc)
# for loading and inspecting the dictionary, go to line 73
#
# %%
# get mrc csv
imag = pd.read_csv('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/Imeagability/mrc-psycholinguistics/mrc2.csv')

# we might not want all the values, so we filter them out
imag = imag[['imag', 'word']].reset_index()
imag['word'] = imag['word'].astype(str)
imag.head()

# %%
imag_dict = {}
for i, r in imag.iterrows():
        
    imag_score = r['imag']
    lex = r['word'].lower()
    lem_sens = lmtzr.lemmatize(lex)

    imag_dict[lem_sens] = {'imag': imag_score}

# get the length of dictionary where entry != 0 imageability
scored_words = {}
for word, attributes in imag_dict.items():
    if attributes['imag'] != 0:
        scored_words[word] = attributes['imag']

print(len(scored_words))

scored_words
# %%
# try out the dict
examples = ['kiss', 'painting', 'hit', 'stone', 'wisdom', 'dog', 'ice', 'unless', 'moral', 'eh', 'honey']
for ex in examples:
    print(ex, imag_dict[ex]['imag'])

# %%
# ok, let's save this as a dictionary
with open('mrc_imageability_only_w_values.json', 'w') as f:
    json.dump(scored_words, f)

# %%
print('All done checking out imageability!')
# %%
