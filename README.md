# Fiction4 sentiment evocation
Data &amp; code for textual features influence on human sentiment perception in literary texts

## 🔬 Data

|             | No. texts | No. annotations   | No. words  | Period     |
|-------------|-----|------|--------|------------|
| **Fairy tales**     | 3   | 772   | 18,597      | 1837-1847  |
| **Hymns**   | 65  | 2,026 | 12,798       | 1798-1873  |
| **Prose**   | 1  | 1,923 | 30,279         | 1952  |
| **Poetry**   | 40  | 1,579 | 11,576         | 1965  |

We present the **Fiction4 corpus** of literary texts, spanning 109 individual texts across 4 genres and two languages (English and Danish) in the 19th and 20th century.
The corpus consists of 3 main authors, Sylvia Plath for poetry, Ernest Hemingway for prose and H.C. Andersen for fairytales. Hymns represent a heterogenous colleciton from Danish official church hymnbooks from 1798-1873.
The corpus was annotated for valence on a sentence basis by at least 2 annotators/sentence.

Full Fiction4 corpus data in `\data\fiction4_data.json`

We compare this fiction corpus again nonfiction texts (across genres)

The nonlit considered is:
1. EmoBank (from this paper [https://aclanthology.org/E17-2092/](https://aclanthology.org/E17-2092/)), repo [here](https://github.com/JULIELab/EmoBank/tree/master). So these are multigenre sentences. (n=10,062 & range=(1 to 674 toks) & mean_length=87.8 toks)
2. Facebook posts (from this paper [https://aclanthology.org/W16-0404.pdf](https://aclanthology.org/W16-0404.pdf)), repo [here](https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal). So these are facebook posts (multiple sentences)(n=2,895 & range=(2 to 445 toks) & mean_length=86.7 toks)

## 💻 Code
All code for our study on human/model sentiment perception across these corpora is available in this repository, see primarily feature extraction (`get_features.py`) and analysis (`analysis.py`).

Annotator agreement calculation for each subcategory of the Fiction4 corpus is in `/annotation/annotator_agreement.py`
