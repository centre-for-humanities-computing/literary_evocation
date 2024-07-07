

# %%
from utils import *
from functions import *

#from functions import plotly_viz_correlation_improved


# %%
# let's see the correlation between the sensorimotor values and the concreteness values
with open("resources/concreteness_brysbaert.json", 'r') as f:
    diconc = json.load(f)

with open("resources/sensorimotor_norms_dict.json", 'r') as f:
    sensori_dict = json.load(f)

# let's see the correlation between the VAD and the concreteness values
with open("resources/vad_lexicon.json", 'r') as f: # note the vad gives valence[0], arousal[1] and dominance[2] for each word
    vad = json.load(f)

with open("resources/mrc_psychol_dict.json", 'r') as f:
    imag = json.load(f)

print('loaded the four lexica')

# %% Convert dictionaries to DataFrames
df_conc = pd.DataFrame.from_dict(diconc, orient='index', columns=['concreteness']).reset_index()
df_conc['word'] = df_conc['index']
df_conc = df_conc [['word', 'concreteness']]
# %%
df_sensori = pd.DataFrame.from_dict(sensori_dict, orient='index').reset_index()
df_sensori['word'] = df_sensori['index']
df_sensori = df_sensori[['word', 'Haptic.mean', 'Interoceptive.mean', 'Visual.mean']]

df_vad = pd.DataFrame.from_dict(vad, orient='index', columns=['valence', 'arousal', 'dominance']).reset_index()
df_vad['word'] = df_vad['index']
df_vad = df_vad[['word', 'arousal', 'dominance']]

df_imag = pd.DataFrame.from_dict(imag, orient='index', columns=['imag']).reset_index()
df_imag['word'] = df_imag['index']
df_imag = df_imag[['imag', 'word']]

dfs = pd.merge(df_imag, df_vad, on='word', how='outer')
dfs = pd.merge(dfs, df_sensori, on='word', how='outer')
dfs = pd.merge(dfs, df_conc, on='word', how='outer')
dfs


# %%
cols = ['imag', 'Haptic.mean', 'Visual.mean']
conc_df = dfs.loc[dfs['concreteness'].notnull()]

plot_scatters(conc_df, cols, 'concreteness', 'lightseagreen', 23, 6, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=True)


# %%
dict_overlap_vad_conc = {}
for word in diconc.keys():
    if word in vad.keys():
        dict_overlap_vad_conc[word] = {'arousal': vad[word][1], 'dominance': vad[word][2], 'concreteness': diconc[word]}


# %%
dict_overlap_sensori_conc = {}
for word in diconc.keys():
    if word in sensori_dict.keys():
        #print(word, diconc[word], sensori_dict[word])
        dict_overlap_sensori_conc[word] = {'Auditory.mean': sensori_dict[word]['Auditory.mean'], 'Gustatory.mean': sensori_dict[word]['Gustatory.mean'], 
                              'Haptic.mean': sensori_dict[word]['Haptic.mean'], 'Interoceptive.mean': sensori_dict[word]['Interoceptive.mean'], 
                              'Olfactory.mean': sensori_dict[word]['Olfactory.mean'], 'Visual.mean': sensori_dict[word]['Visual.mean'], 
                              'concreteness': diconc[word]}

# %%
dict_overlap_imag_conc = {}
for word in diconc.keys():
    if word in imag.keys():
        dict_overlap_imag_conc[word] = {'imageability': imag[word]['imag'], 'concreteness': diconc[word]}


# %% Convert dictionaries to DataFrames
df_vad_conc = pd.DataFrame.from_dict(dict_overlap_vad_conc, orient='index')
df_sensori_conc = pd.DataFrame.from_dict(dict_overlap_sensori_conc, orient='index')
df_imag_conc = pd.DataFrame.from_dict(dict_overlap_imag_conc, orient='index')

# %%
overlap = pd.DataFrame.from_dict(dict_overlap_imag_conc, orient='index', columns=['imageability', 'concreteness']).reset_index()
overlap['word'] = overlap['index']
x = plotly_viz_correlation_improved(overlap, 'concreteness', 'imageability', w=1000, h=650, hoverdata_column='word', canon_col_name='', color_canon=False, save=True)

# %%

#overlap = pd.DataFrame.from_dict(dict_overlap, orient='index', columns=['concreteness', 'sensorimotor']).reset_index()
overlap = pd.DataFrame.from_dict(dict_overlap_sensori_conc, orient='index', columns=['Auditory.mean', 'Gustatory.mean', 'Haptic.mean', 'Interoceptive.mean', 'Olfactory.mean', 'Visual.mean', 'concreteness']).reset_index()
overlap['word'] = overlap['index']

overlap.head()
# %%
filtered_cols = [
 'Haptic.mean',
 'Interoceptive.mean',
 'Visual.mean']

# make scatter subplots
plot_scatters(overlap, filtered_cols, 'concreteness', 'lightseagreen', 20, 6, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=True)
x = plotly_viz_correlation_improved(overlap, 'concreteness', 'Interoceptive.mean', w=1000, h=650, hoverdata_column='word', canon_col_name='', color_canon=False, save=True)

# %%
x = plotly_viz_correlation_improved(overlap, 'concreteness', 'Visual.mean', w=1000, h=650, hoverdata_column='word', canon_col_name='', color_canon=False, save=True)

# %%
def scatterplot_with_annotations(df, x, y, annotate_words):
    plt.figure(figsize=(9, 6), dpi=500)
    
    # Scatter plot all points in df
    plt.scatter(df[x], df[y], alpha=0.2)

    random_state = 32

    # Sample 10 random points from each loc group
    words_high_conc_high_intero = df.loc[(df[x] > 4.3) & (df[y] > 3.2)].sample(n=min(10, len(df.loc[(df[x] > 4.3) & (df[y] > 3.5)])), random_state=random_state)
    words_low_conc_high_intero = df.loc[(df[x] < 1.8) & (df[y] > 3.5)].sample(n=min(10, len(df.loc[(df[x] < 1.8) & (df[y] > 3.5)])), random_state=random_state)
    words_high_conc_low_intero = df.loc[(df[x] > 4.6) & (df[y] < 1.1)].sample(n=min(12, len(df.loc[(df[x] > 4.6) & (df[y] < 1)])), random_state=random_state)
    words_low_conc_low_intero = df.loc[(df[x] < 1.8) & (df[y] < 0.8)].sample(n=min(10, len(df.loc[(df[x] < 1.5) & (df[y] < 1)])), random_state=random_state)

    # Function to avoid overlapping annotations
    def avoid_overlap(annotation, existing_annotations, min_distance=0.1):
        for ann in existing_annotations:
            if np.linalg.norm(np.array(annotation) - np.array(ann)) < min_distance:
                return False
        return True

    # Annotate 10 random points from each group
    annotated_positions = []
    
    for word, x_val, y_val in zip(words_high_conc_high_intero[annotate_words], words_high_conc_high_intero[x], words_high_conc_high_intero[y]):
        # Randomize xytext within a range
        offset_x = np.random.uniform(-0.5, 0.5)
        offset_y = np.random.uniform(-0.5, 0.5)
        annotation_position = (x_val, y_val)
        while not avoid_overlap(annotation_position, annotated_positions):
            annotation_position = (x_val + offset_x, y_val + offset_y)
            offset_x = np.random.uniform(-0.5, 0.5)
            offset_y = np.random.uniform(-0.5, 0.5)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='center')

    for word, x_val, y_val in zip(words_low_conc_high_intero[annotate_words], words_low_conc_high_intero[x], words_low_conc_high_intero[y]):
        offset_x = np.random.uniform(-0.5, 0.5)
        offset_y = np.random.uniform(-0.5, 0.5)
        annotation_position = (x_val, y_val)
        while not avoid_overlap(annotation_position, annotated_positions):
            annotation_position = (x_val + offset_x, y_val + offset_y)
            offset_x = np.random.uniform(-0.5, 0.5)
            offset_y = np.random.uniform(-0.5, 0.5)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='center')

    for word, x_val, y_val in zip(words_high_conc_low_intero[annotate_words], words_high_conc_low_intero[x], words_high_conc_low_intero[y]):
        offset_x = np.random.uniform(-0.5, 0.5)
        offset_y = np.random.uniform(-0.5, 0.5)
        annotation_position = (x_val, y_val)
        while not avoid_overlap(annotation_position, annotated_positions):
            annotation_position = (x_val + offset_x, y_val + offset_y)
            offset_x = np.random.uniform(-0.5, 0.5)
            offset_y = np.random.uniform(-0.5, 0.5)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='right')

    for word, x_val, y_val in zip(words_low_conc_low_intero[annotate_words], words_low_conc_low_intero[x], words_low_conc_low_intero[y]):
        offset_x = np.random.uniform(-0.5, 0.5)
        offset_y = np.random.uniform(-0.5, 0.5)
        annotation_position = (x_val, y_val)
        while not avoid_overlap(annotation_position, annotated_positions):
            annotation_position = (x_val + offset_x, y_val + offset_y)
            offset_x = np.random.uniform(-0.5, 0.5)
            offset_y = np.random.uniform(-0.5, 0.5)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='center')

    # Labels and title
    plt.xlabel(x)
    plt.ylabel(y.split('.')[0])
    #plt.title(f'Scatter Plot of {x} vs {y}')
    plt.legend()  # Show legend with labels

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming 'overlap' is your DataFrame containing the data
# Replace with your actual DataFrame and column names
# Example usage:
scatterplot_with_annotations(overlap, 'concreteness', 'Interoceptive.mean', 'word')


# %%
# checking arousal and dominance as well


dict_overlap
# %%

#overlap = pd.DataFrame.from_dict(dict_overlap, orient='index', columns=['concreteness', 'sensorimotor']).reset_index()
overlap = pd.DataFrame.from_dict(dict_overlap, orient='index', columns=['arousal', 'dominance','concreteness']).reset_index()
#overlap['word'] = overlap['index']

overlap_cols = [x for x in overlap.columns if x != 'index']
for col in overlap_cols:
    overlap[col] = overlap[col].astype(float)

overlap.head()
# %%
overlap_cols

# %%
x = plotly_viz_correlation_improved(overlap, 'concreteness', 'arousal', w=1000, h=650, hoverdata_column='index', canon_col_name='', color_canon=False, save=True)
# %%

def scatterplot_with_annotations(df, x, y, annotate_words):
    plt.figure(figsize=(9, 6), dpi=500)
    
    # Scatter plot all points in df
    plt.scatter(df[x], df[y], alpha=0.2)

    random_state = 32

    # Sample 10 random points from each loc group
    words_high_conc_high_arousal = df.loc[(df[x] > 4.3) & (df[y] > 0.8)].sample(n=min(10, len(df.loc[(df[x] > 4.3) & (df[y] > 0.8)])), random_state=random_state)
    words_low_conc_high_arousal = df.loc[(df[x] < 2) & (df[y] > 0.8)].sample(n=min(10, len(df.loc[(df[x] < 2) & (df[y] > 0.8)])), random_state=random_state)
    words_high_conc_low_arousal = df.loc[(df[x] > 4.6) & (df[y] < 0.3)].sample(n=min(12, len(df.loc[(df[x] > 4.6) & (df[y] < 0.3)])), random_state=random_state)
    words_low_conc_low_arousal = df.loc[(df[x] < 2) & (df[y] < 0.2)].sample(n=min(10, len(df.loc[(df[x] < 2) & (df[y] < 0.2)])), random_state=random_state)

    # Function to avoid overlapping annotations
    def avoid_overlap(annotation, existing_annotations, min_distance=0.1):
        for ann in existing_annotations:
            if np.linalg.norm(np.array(annotation) - np.array(ann)) < min_distance:
                return False
        return True

    # Annotate 10 random points from each group
    annotated_positions = []
    
    for word, x_val, y_val in zip(words_high_conc_high_arousal[annotate_words], words_high_conc_high_arousal[x], words_high_conc_high_arousal[y]):
        offset_x = np.random.uniform(-0.05, 0.05)  # Adjust the range as per your preference
        offset_y = np.random.uniform(-0.05, 0.05)  # Adjust the range as per your preference
        annotation_position = (x_val + offset_x, y_val + offset_y)
        while not avoid_overlap(annotation_position, annotated_positions):
            offset_x = np.random.uniform(-0.05, 0.05)
            offset_y = np.random.uniform(-0.05, 0.05)
            annotation_position = (x_val + offset_x, y_val + offset_y)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='center')

    for word, x_val, y_val in zip(words_low_conc_high_arousal[annotate_words], words_low_conc_high_arousal[x], words_low_conc_high_arousal[y]):
        offset_x = np.random.uniform(-0.05, 0.05)
        offset_y = np.random.uniform(-0.05, 0.05)
        annotation_position = (x_val + offset_x, y_val + offset_y)
        while not avoid_overlap(annotation_position, annotated_positions):
            offset_x = np.random.uniform(-0.05, 0.05)
            offset_y = np.random.uniform(-0.05, 0.05)
            annotation_position = (x_val + offset_x, y_val + offset_y)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='center')

    for word, x_val, y_val in zip(words_high_conc_low_arousal[annotate_words], words_high_conc_low_arousal[x], words_high_conc_low_arousal[y]):
        offset_x = np.random.uniform(-0.05, 0.05)
        offset_y = np.random.uniform(-0.05, 0.05)
        annotation_position = (x_val + offset_x, y_val + offset_y)
        while not avoid_overlap(annotation_position, annotated_positions):
            offset_x = np.random.uniform(-0.05, 0.05)
            offset_y = np.random.uniform(-0.05, 0.05)
            annotation_position = (x_val + offset_x, y_val + offset_y)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='right')

    for word, x_val, y_val in zip(words_low_conc_low_arousal[annotate_words], words_low_conc_low_arousal[x], words_low_conc_low_arousal[y]):
        offset_x = np.random.uniform(-0.05, 0.05)
        offset_y = np.random.uniform(-0.05, 0.05)
        annotation_position = (x_val + offset_x, y_val + offset_y)
        while not avoid_overlap(annotation_position, annotated_positions):
            offset_x = np.random.uniform(-0.05, 0.05)
            offset_y = np.random.uniform(-0.05, 0.05)
            annotation_position = (x_val + offset_x, y_val + offset_y)
        annotated_positions.append(annotation_position)
        plt.annotate(word, annotation_position, textcoords="offset points", xytext=(5,5), ha='center')

    # Labels and title
    plt.xlabel(x)
    plt.ylabel(y.split('.')[0])
    plt.legend()  # Show legend with labels

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming 'df' is your DataFrame containing the data
# Replace with your actual DataFrame and column names
# Example usage:
# scatterplot_with_annotations(df, 'concreteness', 'Interoceptive.mean', 'word')


scatterplot_with_annotations(overlap, 'concreteness', 'arousal', 'index')


