# %%
from utils import *

from functions import *
# %%
# set input path for data
input_path = 'data/hca_edited_data.json'
title = input_path.split('/')[1].split('_')[0]
print('data treated:', title.upper())
# texts should contain sentences and SA scores

# %%
with open(input_path, 'r') as f:
    all_data = json.load(f)

df = pd.DataFrame.from_dict(all_data)
#df.columns = ['ANNOTATOR_1', 'SENTENCE']
print('len data:', len(df))
df.head()

# %%

nan_counts = df.isna().sum()
print("NaN counts per column:")
print(nan_counts)

nan_rows_annotators = df[df[['ANNOTATOR_1', 'ANNOTATOR_2']].isna().any(axis=1)]
print("Rows with NaN values in annotator columns:")
print(nan_rows_annotators)


# %%
# Inter-Annotator Reliability

# Convert annotation columns to float type
df['ANNOTATOR_1'] = df['ANNOTATOR_1'].astype(float)
df['ANNOTATOR_2'] = df['ANNOTATOR_2'].astype(float)
df['ANNOTATOR_3'] = df['ANNOTATOR_3'].astype(float)

# Calculate Spearman correlation between annotators
correlation1, p_value1 = spearmanr(df['ANNOTATOR_1'], df['ANNOTATOR_2'])
print("1-2: IRR: Spearman:", round(correlation1, 3), "p-value:", round(p_value1, 5))

correlation2, p_value2 = spearmanr(df['ANNOTATOR_2'], df['ANNOTATOR_3'])
print("2-3: IRR: Spearman:", round(correlation2, 3), "p-value:", round(p_value2, 5))

correlation3, p_value3 = spearmanr(df['ANNOTATOR_1'], df['ANNOTATOR_3'])
print("1-3: IRR: Spearman:", round(correlation3, 3), "p-value:", round(p_value3, 5))

# Calculate the mean Spearman correlation
mean_corr = (correlation1 + correlation2 + correlation3) / 3
print('Mean Spearman correlation:', round(mean_corr, 3))

# Prepare the annotation data for Krippendorff's alpha calculation
annotations = [df['ANNOTATOR_1'].tolist(), df['ANNOTATOR_2'].tolist(), df['ANNOTATOR_3'].tolist()]

# Calculate Krippendorff's alpha
krip = krippendorff_alpha(annotations)
print("IRR: Krippendorff:", round(krip, 3))

# %%
