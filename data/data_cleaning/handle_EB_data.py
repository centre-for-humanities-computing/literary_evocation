# %%

from utils import *
from functions import *

# %%

eb = pd.read_csv("/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/CHR24/annotation_CHR24/emobank/emobank.csv", index_col=0).reset_index(drop=False)
eb.head()


# %%
# now i want to get the categories the texts belong to
meta = pd.read_csv("/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/CHCAA/FABULANET/CHR24/annotation_CHR24/emobank/EB_meta.tsv", sep='\t')

# %%
# we merge the metadata and the data on the supposed id (id_treated)
merged = eb.merge(meta, how='left', on='id')
len(merged)
# %%
# we drop duplicates
eb_unique = eb.drop_duplicates(subset='text')
print('len of unique sentence:', len(eb_unique))

# %%
merged.head()

# filter cols
eb_df = merged[['V', 'A', 'D', 'text',
       'document', 'category', 'subcategory']].copy()
# %%
# rename columns to match the other datasets
eb_df.columns = ['HUMAN', 'AROUSAL_HUMAN_EB', 'DOMINANCE_HUMAN_EB', 'SENTENCE', 'document', 'category', 'subcategory']
print(len(eb_df))
eb_df.head()

# %%
# Now we can save this to a dict and load it in the analysis script
# dump to json
# eb_dict = eb_df.to_dict(orient='records')
# with open('data/emobank.json', 'w') as f:
#     json.dump(eb_dict, f)

# %%
print('All done checking EmoBank')