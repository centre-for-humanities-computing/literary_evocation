# %%
from utils import *

# %%
with open('/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/DA_literary_SA/resources/mrc_dict_imageability.json', 'r') as f:
    dict_mrc = json.load(f)
# %%

# we want to lemmatize the keys

# Define a function to lemmatize a single key
def lemmatize_key(key):
    return lmtzr.lemmatize(key)

# Define a function to lemmatize all keys of a dictionary
def lemmatize_dict_keys(dictionary):
    lemmatized_dict = {}
    for key, value in dictionary.items():
        lemmatized_key = lemmatize_key(key).lower()
        lemmatized_dict[lemmatized_key] = value
    return lemmatized_dict

# Lemmatize the keys of the dictionary
lemmatized_dict = lemmatize_dict_keys(dict_mrc)
lemmatized_dict['was']['imag']

# imageability goes from 100-?
# Extract 'imag' values from each dictionary
imag_values = [entry['imag'] for entry in lemmatized_dict.values()]

# Calculate maximum and min
max_imag = max(imag_values)
min_imag = min(imag_values)

print("Maximum of 'imag' column:", max_imag)
print("Min of 'imag' column:", min_imag)

# test
lemmatized_dict['ear']

# %%
# dump dict to json
# with open('Resources/mrc_psychol_dict.json', 'w') as f:
#     json.dump(lemmatized_dict, f)
# %%
# %%
print('Imageability dictionary keys lemmatized and the dictionary is checked out!')
# %%
