# %%
# 
from utils import *
from functions import *
# %%

# %%
# Get HCA fairytales
with open('annotation/hca_data.json', 'r') as f:
    hca = json.load(f)

hca = pd.DataFrame.from_dict(hca)
hca['CATEGORY'] = 'fairytales'
print('len data:', len(hca))
hca.head()

# %%
# Get fiction 2 corpus, take only the prose (i.e., hemingway)
with open('annotation/fiction_2.json', 'r') as f:
    f2 = json.load(f)

f2 = pd.DataFrame.from_dict(f2)
f2 = f2.loc[f2['CATEGORY'] == 'prose']
print('len data:', len(f2))

# filter cols
f2 = f2[['avg_concreteness', 'avg_arousal', 'avg_valence', 'avg_dominance',
       'tr_xlm_roberta', 'HUMAN', 'SENTENCE',
       'SENTENCE_ENGLISH', 'id', 'Auditory.mean', 'Gustatory.mean',
       'Haptic.mean', 'Interoceptive.mean', 'Olfactory.mean', 'Visual.mean',
       'CATEGORY', 'vader', 'ANNOTATOR_1', 'ANNOTATOR_2', 'concreteness', 'avg_imageability']].copy()
f2.tail()

 #%%
 # Get the hymns
with open('annotation/hymns_data.json', 'r') as f:
    hymns = json.load(f)

hymns = pd.DataFrame.from_dict(hymns)
print('len data:', len(hymns))
hymns.head()

# %%
# Get the poems (Plath)
with open('annotation/plath_data.json', 'r') as f:
    plath = json.load(f)

plath = pd.DataFrame.from_dict(plath)
plath['CATEGORY'] = 'poetry'
print('len data:', len(plath))
plath.head()

# %%
# Merge these
merged = pd.concat([hca, plath, f2, hymns]).reset_index(drop=True)
len(merged)
# %%
merged.head()

# %%
# Get the length of each genre
for cat in merged['CATEGORY'].unique():
    print(cat, len(merged.loc[merged['CATEGORY'] == cat]))
# %%
# Check for Nan values
nan_counts = merged.isna().sum()
print("NaN counts per column:")
print(nan_counts)

nan_rows_annotators = merged[merged[['HUMAN', 'tr_xlm_roberta']].isna().any(axis=1)]
print("Rows with NaN values in SA columns:")
print(nan_rows_annotators)

# %%
# save full ficton4 to json
# with open('data/fiction4.json', 'w') as f:
#     json.dump(merged.to_dict(), f)
# %%
