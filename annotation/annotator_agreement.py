# %%
from utils import *
from functions import *
# %%
# set input path for data
input_path = 'annotation/hca_data.json' # HCA

title = input_path.split('/')[1].split('_')[0]
print('data treated:', title.upper())
# texts should contain sentences and SA scores for each annotator

# %%
# load and make df
with open(input_path, 'r') as f:
    all_data = json.load(f)
df = pd.DataFrame.from_dict(all_data)
print('len data:', len(df))
df.head()

# if getting hemingway, we want to take hemingway from the bigger fiction2 corpus
if title == 'fiction':
    print('Getting Heminway data from bigger corpus')
    # filter df
    df = df.loc[df['CATEGORY'] == 'prose']
    print('len filtered data:', len(df))
    # overwrite title
    title = 'hemingway'
    print(title.upper())
    df.head()

# %%
# see the columns indexed and check for NaN values in the annotator-columns
# check how many annotators there are
no_annotators = [x for x in df.columns if x.startswith('ANNOTATOR')]
print(f'Number of annotators for {title.upper()}:', len(no_annotators))

nan_counts = df.isna().sum()
print("NaN counts per column:")
print(nan_counts)

nan_rows_annotators = df[df[no_annotators].isna().any(axis=1)]
print("Rows with NaN values in annotator columns:")
print(nan_rows_annotators)

# %%
# Inter-Annotator Reliability
# Convert annotation columns to float type
for col in no_annotators:
    df[col] = df[col].astype(float)

# define loop to print correlations between all annotators
def get_print_spearman(df, annotator_cols):
    num_annotators = len(annotator_cols)

    saved_corrs = []

    for i in range(num_annotators):
        for j in range(i + 1, num_annotators):
            correlation, p_value = spearmanr(df[annotator_cols[i]], df[annotator_cols[j]])
            print(f"Annotators {i} and {j}: IRR: Spearman: {round(correlation, 3)}, p-value: {round(p_value, 5)}")
            saved_corrs.append(correlation)

    # and print the average correlation (if there are more than 2)
    if num_annotators > 2:
        print("Average IRR: Spearman:", round(sum(saved_corrs) / num_annotators,3))

    # and print the krippendorff's alpha between annotators
    listed_annotations = df[annotator_cols].T.values.tolist()
    krip = krippendorff_alpha(listed_annotations)
    print("IRR: Krippendorff:", round(krip, 3))

get_print_spearman(df, no_annotators)

# %%
print(f'Checked IRR for: {title.upper()}')

# %%
