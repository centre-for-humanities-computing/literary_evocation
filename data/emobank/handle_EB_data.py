# %%

from utils import *
from functions import *

# %%

eb = pd.read_csv("/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/CHR24/annotation_CHR24/emobank/emobank.csv", index_col=0).reset_index(drop=False)
#eb = pd.read_csv("data/emobank_readers_scores.csv", index_col=0).reset_index(drop=False)
# %%
eb.head()

# so i donøt know how the id columns correspond to the meta data, so i will try to match them
# somehow on the id last digits....
# %%
# # get the last digits in 'id' to get id to merge on with metadata
# eb['id_treated'] = eb['id'].apply(lambda x: '_'.join(x.split('_')[-2:]))
# eb.head()

# %%
# now i want to get the categories the texts belong to
meta = pd.read_csv("/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/CHR24/annotation_CHR24/emobank/EB_meta.tsv", sep='\t')
# get the last digits in 'id' to get id to merge on
# meta['id_treated'] = meta['id'].apply(lambda x: '_'.join(x.split('_')[-2:]))
# meta['id_treated'].value_counts()

# %%
# check len of unique id sequences
# unique_sequences = meta['id_treated'].apply(lambda x: tuple(x)).unique()
# len([list(seq) for seq in unique_sequences])

# #len(meta)

# %%
# we merge the metadata and the data on the supposed id (id_treated)
merged = eb.merge(meta, how='left', on='id')
# make dataframe analysed (eb) into the merged one now
# eb = merged.copy()
# len(eb)

# %%
# duplicate maskl
eb_dup_mask = merged["text"].duplicated()
ep_dupes = eb[eb_dup_mask]

# %%
len(merged)
# %%
# we drop duplicates
eb_unique = eb.drop_duplicates(subset='text')
print('len of unique sentence:', len(eb_unique))

# %%
merged.head()
eb_df = merged[['V', 'A', 'D', 'text',
       'document', 'category', 'subcategory']].copy()
# %%
# rename columns to match the other datasets
eb_df.columns = ['HUMAN', 'AROUSAL_HUMAN_EB', 'DOMINANCE_HUMAN_EB', 'SENTENCE', 'document', 'category', 'subcategory']
eb_df.head()
#%%
len(eb)
# %%
# Now we can save this to a dict and load it in the analysis script
# dump to json
eb_dict = eb_df.to_dict(orient='records')
with open('data/emobank_raw.json', 'w') as f:
    json.dump(eb_dict, f)

# %%


# with open('data/emobank_w_features_and_cats.json', 'r') as f:
#     all_data = json.load(f)

# df = pd.DataFrame.from_dict(all_data)


# # %%
# eb_dict = df.to_dict(orient='records')
# with open('data/emobank_w_features_and_cats.json', 'w') as f:
#     json.dump(eb_dict, f)
# %%
