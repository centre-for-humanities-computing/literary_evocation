
"""
main script to analyse datasets for evocative sentiment expression strategies
we analyse each corpus seperately

Procedure
- Getting data ready
- Data statistics
- Part 1: comparing explicit/implicit groups of sentences
- Part 2: correlation of feature values with absolute disagreement between human/SA model
- Lastly feature values and overall human/model correlation for the corpus treated
"""
# %% 
from utils import *
from functions import *

# %%
# we define some presets, where to store images, 
# which corpus is treated, with the necessary changes to column names
# whether to filter the data or not (default, not).

# set out path for visualizations
output_path = 'figures/'
# set input path for data
input_path =  'data/fiction4_data.json'
# make a save-title
save_title = input_path.split('/')[-1].split('_')[0]

# choose whether or not to filter the whole data by sentence length (we ended up not doing this)
filter = False
len_threshold = 5

if filter == True:
    print('data treated:', save_title, '-- filtered for length == True')
    save_title += '_filtered'
else:
    print('dataset treated:', save_title.upper())

# categories are in differently named columns in each dataset, so we use a map
column_map = {'emobank': 'category','fiction3':'CATEGORY', 'fiction4': 'CATEGORY'}

# %%
# get data
# open the json data and make df
with open(input_path, 'r') as f:
    all_data = json.load(f)
data = pd.DataFrame.from_dict(all_data)

# if FB data, add the human annotated arousal
if save_title == 'FB':
    df['avg_harousal'] = df['harousal']

# if emobank, filter out semeval
if save_title == 'emobank':
    data = data.loc[data['category'] != 'SemEval']
    # and add the human annotated arousal
    data['avg_harousal'] = data['AROUSAL_HUMAN_EB']
print(f'len {save_title.upper()}', len(data))
data.head()

# %%
# Tokenize and get len of sentences/posts
data['SENTENCE_TOKENNIZED'] = data['SENTENCE'].apply(lambda x: nltk.wordpunct_tokenize(x.lower()))
lens = data['SENTENCE_TOKENNIZED'].apply(lambda x: len(x))
data['SENTENCE_LENGTH'] = lens
print('sentences tokenized')

# filtering on sentence length if set to True
if filter == True:
    df = data.loc[data['SENTENCE_LENGTH'] > len_threshold].reset_index(drop=True)
    print(f'data is FILTERED, sentences < len {len_threshold} are removed')
    print('len original/len filtered', len(data), len(df))
else:
    df = data
    print('data is UNFILTERED')
    print('len data:', len(df))

# %%
# If treating EmoBank, clean the data further and print the numbers for each category too
if 'emobank' in save_title:
    # get rid of the SemEval "genre"
    df = df.loc[df['category'] != 'SemEval']
    print('SemEval category removed from EmoBank')
    print('\n')

    for cat in df['category'].unique():
        cat_df = df.loc[df['category'] == cat]
        print(f'{cat.upper()} avg words per sentence/post:', round(cat_df['SENTENCE_LENGTH'].mean(),1), '--std:', round(cat_df['SENTENCE_LENGTH'].std(),1))
        print(f'{cat.upper()} number of sentences/posts:', len(cat_df))
        print(f'{cat} number of words:', cat_df['SENTENCE_LENGTH'].sum())

# data statistics for full corpus
print('avg words per sentence/post:', round(df['SENTENCE_LENGTH'].mean(),1), '--std:', round(df['SENTENCE_LENGTH'].std(),1))
print('number of sentences/posts:', len(df))
print('number of words:', df['SENTENCE_LENGTH'].sum())

# if we are treating the Ficiton4 corpus, get some numbers on each category as well
if save_title == 'fiction4':
    data['CATEGORY'].value_counts()

    for cat in data['CATEGORY'].unique():
        print(cat, '--')
        counts_df = df.loc[data['CATEGORY'] == cat]
        s_len = counts_df['SENTENCE_LENGTH'].mean()
        no_words = counts_df['SENTENCE_LENGTH'].sum()
        lens = len(counts_df)
        print(cat, 'len data:', lens, 'avg s_len:', s_len, 'no words:', no_words)

# Rename the sensorimotor columns (just for layout)
df['avg_visual'] = df['Visual.mean']
df['avg_haptic'] = df['Haptic.mean']
df['avg_interoceptive'] = df['Interoceptive.mean']

# %%
# we want to normalize the dictionary scores before using it to filter out the groups for certain datasets
# adjust human range if using these datasets
data_to_normalize = ['emobank', 'FB']

if save_title in data_to_normalize:
    df['HUMAN'] = normalize(df['HUMAN'], scale_zero_to_ten=True) # we scale it 0-10 to get it comparable to the Fiction4 corpus
    df.head()

    print(f'{save_title.upper()} avg human valence scores were normalized to range 0-10')

# %%
# Visualize SA
# get a distribution of human / tranformer scores
dist_data = df
sns.set_style('whitegrid')
res = plot_kdeplots_or_histograms(dist_data, ['HUMAN', 'tr_xlm_roberta'], 'histplot', '',2, l=14, h=3)

# %%
# EXPERIMENT 1
print('EXPERIMENT 1')
print('Comparing two groups')
# Here we make two groups, one of "implicit" and one of "explicit" sentiment, signifying sentences where human/model diverge/converge

# First, we filter out sentences with lukewarm human ratings (neutral scores)
# human scale is 0-10
filtered = df.loc[(df['HUMAN'] <= 4.5) | (df['HUMAN'] >= 5.5)].reset_index(drop=False)
print('len filtered:', len(filtered))

# then we set the (absolute) threshold for when the model scores something neutral
# model scale is -1 to 1
threshold = 0.1

# make implicit group
implicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) <= threshold)]
print('len IMplicit group:', len(implicit_df))

# make explicit group
explicit_df = filtered.loc[(abs(filtered['tr_xlm_roberta']) > threshold)]
print('len EXplicit group:', len(explicit_df))

# %% make fig of groups & sizes
implicit_df['GROUP'] = 'implicit'
explicit_df['GROUP'] = 'explicit'
filtered['GROUP'] = 'subset'
df['GROUP'] = 'all'
dfs = pd.concat([implicit_df, explicit_df, filtered, df])

# Aggregate to get counts for each group
group_counts = dfs['GROUP'].value_counts().reset_index()
group_counts.columns = ['GROUP', 'Counts']

# Define colors for each group
colors = ['#f77f00', '#2a9d8f', 'lightgrey']
colors = ['lightgrey', 'darkgrey'] + sns.color_palette("rocket", n_colors=2)

plt.figure(figsize=(8, 3), dpi=300)
bars = plt.barh(group_counts['GROUP'], group_counts['Counts'], color=colors)

# Annotate bars with counts
for bar in bars:
    width = bar.get_width()
    plt.annotate('{}'.format(width),
                 xy=(width / 2, bar.get_y() + bar.get_height() / 2),
                 xytext=(3, 0),  # 3 points horizontal offset
                 textcoords="offset points",
                 ha='center', va='center')
plt.xlabel('')
plt.show()

# %%
# Inspecting groups in terms of feature values
# we define the measures we want to use
measure_list = ['avg_arousal', 'avg_dominance', 'avg_concreteness', 
                'avg_imageability', 'avg_visual', 'avg_haptic', 'avg_interoceptive'] 

# and use human arousal and dominance if treating EmoBank
if save_title == 'emobank':
    measure_list = ['avg_harousal'] + measure_list
    print('EmoBank avg human dominance & arousal is also used')

# and use human arousal if treating FB
if save_title == 'FB':
    measure_list = ['avg_harousal'] + measure_list 
    print('FB avg human dominance & arousal is also used')

print('measures considered:', measure_list)

# %%
# We do a Mann-Whitney U ranks test (we don't want to assume normal distributions), 
# with the option of checking a permutation test as well for good measure (uncomment below if wanted)

    # Permutation test
def mean_diff(x, y):
    return np.mean(x) - np.mean(y)

ustats = []
pvals = []
perm_stats = []
perm_pvals = []

for measure in measure_list:
    df1 = explicit_df.loc[explicit_df[measure].notnull()]
    df2 = implicit_df.loc[implicit_df[measure].notnull()]
    print()
    print('groupsizes', len(df2), len(df1))
    values1, values2 = df1[measure], df2[measure]

    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
    ustats.append(u_stat)
    pvals.append(p_value)

    # perm_result = permutation_test((values1, values2), mean_diff, vectorized=False, n_resamples=100000, alternative='two-sided')
    # perm_stats.append(perm_result.statistic)
    # perm_pvals.append(perm_result.pvalue)
    
    print(measure.upper(), "U statistic:", u_stat, "P-value:", np.round(p_value, 5))
    #print(f'Permutation Test Statistic: {perm_result.statistic}, P-value: {np.round(perm_result.pvalue, 5)}')
    print()


# %%
# Boxplots
implicit_df['GROUP'] = 1
explicit_df['GROUP'] = 0

df1 = explicit_df.loc[explicit_df['avg_valence'].notnull()]
df2 = implicit_df.loc[implicit_df['avg_valence'].notnull()]

both_groups = pd.concat([df1, df2])

sns.set_style("whitegrid")
x = pairwise_boxplots_canon(both_groups, measure_list, category='GROUP', category_labels=['implicit', 'explicit'], 
                            plottitle=save_title.upper(), outlier_percentile=100, remove_outliers=False, h=9, w=20, save=True)


# %%

# plot the CEDs and do the kolmogorov-smirnov test
sensorimotor = ['Auditory.mean', 'Gustatory.mean', 'Olfactory.mean']
ced_plot(implicit_df, explicit_df, measure_list, measure_list, save=True, save_title=save_title)
ced_plot(implicit_df, explicit_df, sensorimotor, sensorimotor, save=True, save_title=save_title + '_sensorimotor')
# 

# %%
# histplots, a similar visualization of distribution
print(f'whole {save_title.upper()}')
histplot_two_groups(implicit_df, explicit_df, measure_list, measure_list, l=40, h=4, title_plot=f"{save_title.split('_')[0]} All texts", density=True, save=True, save_title=save_title)

# again, just divided into two plots
histplot_two_groups(implicit_df, explicit_df, measure_list[:3], measure_list[:3], l=18, h=3, title_plot="", density=True, save=True, save_title=save_title)
histplot_two_groups(implicit_df, explicit_df, measure_list[3:], measure_list[3:], l=20, h=3.1, title_plot="", density=True, save=True, save_title=save_title)

# %%
# if there are categories in the data, we want to show a histplot of each category seperately.
if save_title in column_map.keys():
    categories = df[column_map[save_title]].unique()

    for cat in categories:
        implicit_df_cat = implicit_df.loc[implicit_df[column_map[save_title]] == cat]
        explicit_df_cat = explicit_df.loc[explicit_df[column_map[save_title]] == cat]
        print(f'{cat.upper()}: Groups: len implicit/explicit:', len(implicit_df_cat), '/', len(explicit_df_cat))
        histplot_two_groups(implicit_df_cat, explicit_df_cat, measure_list, measure_list, l=50, h=5, title_plot=cat, density=True, save=False, save_title=save_title + '_' + cat)

        # and get the test for each subgroup
        ustats = []
        pvals = []

        for measure in measure_list:
            df1 = explicit_df_cat.loc[explicit_df_cat[measure].notnull()]
            df2 = implicit_df_cat.loc[implicit_df_cat[measure].notnull()]
            print()
            print('groupsizes', len(df2), len(df1))
            values1, values2 = df1[measure], df2[measure]

            u_stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
            ustats.append(u_stat)
            pvals.append(p_value)

            print(cat.upper(), measure.upper(), "U statistic:", u_stat, "P-value:", np.round(p_value, 5))
            print()

# %%
# we also confirm the overall (not subgroup) results with a linear regression (trying to predict group)

out = {}
for measure in measure_list:
    filt = both_groups.loc[both_groups[measure].notnull()]
    X = filt[measure]
    y = filt['GROUP']

    lm = pg.linear_regression(X, y)

    temp = lm[["coef", "r2", "se", "adj_r2", "pval"]].iloc[1, :].to_dict()
    temp = {key: round(value, 3) for key, value in temp.items()} # round

    get_y_pred = pg.linear_regression(X, y, as_dataframe=False)
    
    pred = list(get_y_pred['pred'])
    #temp['RMSE'] = root_mean_squared_error(y, pred, squared=False) #True returns MSE
    out[measure] = temp

for measure in measure_list:
    print(measure, '\n', out[measure])

print('\n', 'Experiment 1 done')
# %%
# EXPERIMENT 2
print('EXPERIMENT 2')
# we want to see if there is a correlation between the Human/Model absolute difference and the features overall

# Normalize the human values to the same scale as the model (-1 to 1)
df['HUMAN_NORM'] = normalize(df['HUMAN'], scale_zero_to_ten=False)

# get the absolute Human/Model difference
df['ROBERTA_HUMAN_DIFF'] = abs(abs(df['HUMAN_NORM']) - abs(df['tr_xlm_roberta']))

# we want to inspect the relation between each measure and Human/Model difference doing a scatterplot
print(f'All genres -- {measure} corr w. disagreement')
for measure in measure_list:
    x = plotly_viz_correlation_improved(df, measure, 'ROBERTA_HUMAN_DIFF', w=500, h=200, hoverdata_column='SENTENCE', canon_col_name='', color_canon=False, save=False)

# %%
# we want to try and see if the correlation improves at differente thresholds of sentence length
# the whole data (not divided into categories)
# and visualizing it
thresholds = [0, 5]

# going back to the original (certainly) unfiltered dataframe -- 'data'
df['HUMAN_NORM'] = normalize(df['HUMAN'], scale_zero_to_ten=False)
df['ROBERTA_HUMAN_DIFF'] = abs(abs(df['HUMAN_NORM']) - abs(df['tr_xlm_roberta']))

# correlation at different thresholds per each category
# this is only for the data with categories
if save_title in column_map:

    # Let's try filtering for the different categories and correlating diff score to the features
    categories = df[column_map[save_title]].unique()
    category_data_all = {}

    for category in categories:
        print(category)
        category_data_per_threshold = {}

        category_df = df.loc[df[column_map[save_title]] == category]

        for threshold in thresholds:
            print('\n')
            category_data_threshold = {}

            # Filter data based on category and threshold
            data_filtered_for_s_len = category_df.loc[category_df['SENTENCE_LENGTH'] > threshold]

            # Drop NaNs before correlation
            data_filtered_for_s_len_dropna = data_filtered_for_s_len.dropna(subset=['ROBERTA_HUMAN_DIFF', 'avg_concreteness', 'avg_arousal', 'avg_imageability'])

            measure_results = {}

            for measure in measure_list:
                # Calculate correlation for each measure
                corr = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna[measure])
                measure_results[measure] = {'correlation': round(corr[0], 3), 'p-value': round(corr[1], 5)}
                print(category, 'T:', threshold, measure, 'correlation::', round(corr[0], 3), 'p', round(corr[1], 5))
        print('\n')

# %%
# We just want the overall correlation of the whole data with the 5 word sentence threshold
threshold = 5
data_filtered_for_s_len = df.loc[df['SENTENCE_LENGTH'] > threshold]

# drop NaNs before correlation
data_filtered_for_s_len_dropna = data_filtered_for_s_len.dropna(subset=['ROBERTA_HUMAN_DIFF', 'avg_concreteness', 'avg_arousal', 'avg_imageability'])
print('len data', len(data_filtered_for_s_len_dropna))

print(f'correlation of whole data with word/sentence threshold of: {threshold}')

for measure in measure_list:
    correlation_results = stats.spearmanr(data_filtered_for_s_len_dropna['ROBERTA_HUMAN_DIFF'], data_filtered_for_s_len_dropna[measure])
    corr_value = round(correlation_results[0], 3)
    p_value = round(correlation_results[1], 5)
    print('correlation::', measure, corr_value, p_value)


# %%
# Feature values
# and get the mean, std, median for the features across the categories
features = measure_list + ['ROBERTA_HUMAN_DIFF']
data_unfiltered = {}

print('All data')
for feature in features:
            # get mean, std, median
        mean = round(df[feature].mean(), 3)
        std = round(df[feature].std(), 3)
        data_unfiltered[feature] = {'mean': mean, 'std': std}#, 'median': median}

print(pd.DataFrame.from_dict(data_unfiltered))


category_data_unfiltered = {}

if save_title in column_map:
    categories = df[column_map[save_title]].unique()

    for category in categories:
        category_data_feature = {}
        category_df = df.loc[df[column_map[save_title]] == category]
        
        # We loop through each category
        for feature in features:
            # Get mean, std, median
            mean = round(category_df[feature].mean(), 3)
            std = round(category_df[feature].std(), 3)
            median = round(category_df[feature].median(), 3)
        
            category_data_feature[feature] = {'mean': mean, 'std': std, 'median': median}
        
        category_data_unfiltered[category] = category_data_feature
    print('Data per category')
    print(pd.DataFrame.from_dict(category_data_unfiltered))


# %%
# Lastly, we want the overall correlation human/model in the corpus treated
print('correlations between human/roberta')

# We want to get the correlation between RoBERTa and Humans for each of our datasets
# get the overall correlation between human/vader
df = df.loc[df['HUMAN'].notnull()]
print(len(df))
correlation_results = stats.spearmanr(df['tr_xlm_roberta'], df['HUMAN'])
corr_value = round(correlation_results[0], 3)
p_value = round(correlation_results[1], 5)
print('correlation, overall, human vs roberta :', corr_value, p_value)

# for the datasets that have different categories, we want to check the correlation for each category with the human mean
map_categories = {'emobank': 'category', 'fiction4': 'CATEGORY'}

if save_title in map_categories:
    for cat in df[map_categories[save_title]].unique():
        dat = df.loc[df[map_categories[save_title]] == cat]

        correlation_results = stats.spearmanr(dat['tr_xlm_roberta'], dat['HUMAN'])
        corr_value = round(correlation_results[0], 3)
        p_value = round(correlation_results[1], 5)
        print(f'correlation, human vs roberta for cat: {cat}', corr_value, p_value)

# %%

print('Analysis all done!')
