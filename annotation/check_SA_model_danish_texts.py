
# %% 
from utils import *
from functions import *


# open the merged json and make df
with open('annotation/hca_data.json', 'r') as f:
    all_data = json.load(f)
hca = pd.DataFrame.from_dict(all_data)

with open('annotation/hymns_data.json', 'r') as f:
    all_data = json.load(f)
hymns = pd.DataFrame.from_dict(all_data)

data = pd.concat([hca, hymns])
data = data[['HUMAN', 'SENTENCE',
       'SENTENCE_ENGLISH', 'tr_xlm_roberta']]
print(len(data))
data.tail()
# %%
# for the texts originally in Danish, check the difference between applying roberta in each language

# first print the correlation of the SA on English translations (already computed)
correlation_results = stats.spearmanr(data['tr_xlm_roberta'], data['HUMAN'])
corr_value = round(correlation_results[0], 3)
p_value = round(correlation_results[1], 5)
print('correlation, human vs roberta HCA but SA on ENGLISH translations:')
print(corr_value, p_value)


# %%
# then we want to redo the roberta values directly on the danish texts
xlm_model = pipeline(model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

xlm_labels = []
xlm_scores = []

for s in data['SENTENCE']:
    # Join to string if list
    if isinstance(s, list):
        s = " ".join(s)
    # get sent-label & confidence to transform to continuous
    sent = xlm_model(s)
    xlm_labels.append(sent[0].get("label"))
    xlm_scores.append(sent[0].get("score"))

# function defined in functions to transform score to continuous
xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])
data["tr_xlm_roberta"] = xlm_converted_scores

# see correlation with human values
correlation_results = stats.spearmanr(data['tr_xlm_roberta'], data['HUMAN'])
corr_value = round(correlation_results[0], 3)
p_value = round(correlation_results[1], 5)
print('correlation, human vs roberta (SA on DANISH original texts):')
print(corr_value, p_value)

# %%
# and try to do it on google translations to compare the performance (we did this before we decided to go 
# through and revise the translations manually)

# get the English sentences
from deep_translator import GoogleTranslator

sents = list(data['SENTENCE'])

# Translating all sentences
# This takes some time to run
translated_sents = []

for sent in sents:
    translated = GoogleTranslator(source='da', target='en').translate(sent)
    translated_sents.append(translated)

print('len en sents:', len(translated_sents))

data['GOOGLE_SENTS'] = translated_sents

xlm_labels = []
xlm_scores = []

for s in data['GOOGLE_SENTS']:
    # Join to string if list
    if isinstance(s, list):
        s = " ".join(s)
    # get sent-label & confidence to transform to continuous
    sent = xlm_model(s)
    xlm_labels.append(sent[0].get("label"))
    xlm_scores.append(sent[0].get("score"))

# function defined in functions to transform score to continuous
xlm_converted_scores = conv_scores(xlm_labels, xlm_scores, ["positive", "neutral", "negative"])
data["tr_xlm_roberta"] = xlm_converted_scores

# see correlation with human values
correlation_results = stats.spearmanr(data['tr_xlm_roberta'], data['HUMAN'])
corr_value = round(correlation_results[0], 3)
p_value = round(correlation_results[1], 5)
print('correlation, human vs roberta (SA on DANISH google translated texts):')
print(corr_value, p_value)

# %%

print('Checked corr between human/roBERTa on Danish, google translated Danish and manually checked English translations')