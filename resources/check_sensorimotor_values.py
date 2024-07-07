
# %%
from utils import *
from functions import *

# %%

# Script for making the lancaster sensorimotor norms into a dictionary (lemmatizing etc)
# for loading and inspecting the dictionary, go to line 73
#
# %%
# get sensorimotor csv
sensori = pd.read_csv('resources/Sensorimotor_norms_21Mar2024.csv')
print('loaded sensorimotor, len:', len(sensori['Word'].keys()))

sensori.head()
# %%

mean_cols = [x for x in sensori.columns if x.endswith('mean')]
mean_cols

# %%
# we do not want all the sensorimotor values, so we filter them out
filtered_cols = ['Auditory.mean',
 'Gustatory.mean',
 'Haptic.mean',
 'Interoceptive.mean',
 'Olfactory.mean',
 'Visual.mean']
#  'Foot_leg.mean',
#  'Hand_arm.mean',
#  'Head.mean',
#  'Mouth.mean',
#  'Torso.mean']

# %%
# 5%
sensori_dict = {}
for i, r in sensori.iterrows():
    values_all_sens = []
    for col in filtered_cols:
        auditory, gustatory, haptic, interoceptive, olfactory, visual = r['Auditory.mean'], r['Gustatory.mean'], r['Haptic.mean'], r['Interoceptive.mean'], r['Olfactory.mean'], r['Visual.mean']

        lex = r['Word'].lower()
        lem_sens = lmtzr.lemmatize(lex)
        #values_all_sens.append(r[col]) # for taking all

    sensori_dict[lem_sens] = {'Auditory.mean': auditory, 'Gustatory.mean': gustatory, 'Haptic.mean': haptic, 'Interoceptive.mean': interoceptive, 'Olfactory.mean': olfactory, 'Visual.mean': visual}

# %%
# try out the sensori dict
print(filtered_cols)
examples = ['kiss', 'attack', 'hit', 'thought', 'wisdom', 'dog', 'ice', 'unless', 'moral', 'eh', 'honey']
for ex in examples:
    print(ex, sensori_dict[ex])

# %%
# ok, let's save this as a dictionary
# with open('resources/sensorimotor_norms_dict.json', 'w') as f:
#     json.dump(sensori_dict, f)

# %%
print('All done checking out Lancaster Sensorimotor Norms!')
# %%
