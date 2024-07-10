# %%
from utils import *
from functions import *

# %%
sentence = "Ay he said aloud there is no translation for this word and perhaps it is just a noise such as a man might make involuntarily feeling the nail go through his hands and into the wood"
print('senstence:', sentence)
# What does this sentence get in scoring from model/human?
# index 4368 in fiction 2

with open('annotation/fiction_2.json', 'r') as f:
    data = json.load(f)

print('human score:', data['HUMAN']['4368'])
print('model score:', data['tr_xlm_roberta']['4368'])
# human score corresponds to -0.5 on the -1 to 1 scale
# %%
# tokenize and lemmatize
toks = nltk.wordpunct_tokenize(sentence.lower())
lems = [lmtzr.lemmatize(word) for word in toks]

# %%
# loading dicts, getting feature values
print('# PART 1: loading dicts, getting feature values')

# Loading concreteness lexicon
# the json is structured so that the word is the key, the value the concreteness score
with open("resources/concreteness_brysbaert.json", 'r') as f:
    diconc = json.load(f)
print('loaded concreteness lexicon, len:', len(diconc))

# loading VAD
# same here, where values are the valence, arousal and dominance scores (in that order)
with open("resources/VAD_lexicon.json", 'r') as f:
    dico = json.load(f)
print('loaded VAD lexicon, len:', len(dico))

# reopen save dict of sensorimotor values
with open('resources/sensorimotor_norms_dict.json', 'r') as f:
    sensori_dict = json.load(f)
print('loaded sensorimotor lexicon, len:', len(sensori_dict))

# and get the imageability dict from MRC psycholinguistics database
with open('resources/mrc_psychol_dict.json', 'r') as f:
    dict_mrc = json.load(f)
print('loaded imageability lexicon, len:', len(dict_mrc))


# %%

dict_feats = {}

for word in lems:
    # Initialize the feature dictionary for each word
    features = {}
    
    # Check and add concreteness
    if word in diconc.keys():
        features['concreteness'] = diconc[word]
    else:
        features['concreteness'] = np.nan
    
    # Check and add imagery
    if word in dict_mrc.keys() and 'imag' in dict_mrc[word]:
        features['imageability'] = dict_mrc[word]['imag'] / 100 # get the imag scores to be smaller
    else:
        features['imageability'] = np.nan
    
    # arousal
    if word in dico.keys():
        features['arousal'] = dico[word][1]
        features['dominance'] = dico[word][2]
    else:
        features['arousal'] = np.nan
        features['dominance'] = np.nan

    # sensori
    if word in sensori_dict.keys():
        features['haptic'] = sensori_dict[word]['Haptic.mean']
        features['interoceptive'] = sensori_dict[word]['Interoceptive.mean']
        features['visual'] = sensori_dict[word]['Visual.mean']
    else:
        features['haptic'] = np.nan
        features['interoceptive'] = np.nan
        features['visual'] = np.nan
    
    # Add the features dictionary to the main dictionary
    dict_feats[word] = features


# %%
dict_feats
df = pd.DataFrame.from_dict(dict_feats, orient='index')
df.head(10)

# %%
# normalize all values to the range 0-1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# for each column individually
df_normalized = df.copy()

for column in df.columns:
    # Only scale if column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        # Reshape the column to 2D array for the scaler
        col_values = df[[column]].values
        # Apply Min-Max scaling, ignore NaNs during fitting and transform
        non_nan_values = col_values[~np.isnan(col_values)].reshape(-1, 1)
        col_scaled = scaler.fit_transform(non_nan_values)
        # Create a new column with scaled values and restore NaNs
        col_normalized = np.full_like(col_values, np.nan)
        col_normalized[~np.isnan(col_values)] = col_scaled.flatten()
        df_normalized[column] = col_normalized

df_normalized.head(10)

# %%
# Set the figure size
plt.figure(figsize=(26, 5))

# Color palette
palette = sns.color_palette('tab10', len(df.columns))

# Plot each column separately
for i, column in enumerate(df_normalized.columns):
    values = pd.to_numeric(df_normalized[column], errors='coerce')  # Convert to numeric
    x = df_normalized.index.values
    
    sns.pointplot(x=x, y=values, color=palette[i], markers='.', label=column, linewidth=8, alpha=0.5)

plt.xlabel('t')
plt.ylabel('Values')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=len(df_normalized.columns))
plt.xticks(rotation=30)
plt.show()

# %%
# get the mean values for cols
for col in df_normalized.columns:
    vals = df_normalized[col].astype(float).fillna(0)
    mean = round(vals.mean(), 2)
    print(col, mean)
# %%
print('all done checking example')
# %%
