# Project Documentation: Researchly Expert System

Repository layout: `data/` (datasets), `notebooks/` (one per question), `deployment/` (web app), `docs/` (this file and the
interactive topic map `lda_intertopic_map.html`). Notebooks are saved with their outputs and charts.

---

## Question 1: Research Papers - Expert System (`notebooks/Q1_Preprocessing_and_Baseline.ipynb`)

### 1.1 Load and denoise
`research_abstracts.json` has 1,138 rows with columns `DOI`, `Abstract` and `Label` (`ai` or `classical`). 13 duplicate
abstracts were removed (1,125 remain: 890 classical, 235 AI), so the same text cannot appear in both the train and the test set.
Denoising (`deployment/text_utils.py`): lowercase; remove URLs, e-mails, HTML tags, numbers and punctuation; remove English
stop words; drop 1 to 2 letter tokens. The median abstract goes from 152 to 89 words. The cleaned data frame is saved as
`data/abstracts_clean.csv`.

### 1.2 Columns and feature engineering
`DOI` is an identifier with no topic meaning (and 4 missing values), so it is dropped. `Abstract` (cleaned) is the feature
source and `Label` is the target. Features are **TF-IDF** vectors (5,073 terms with `min_df=2`). Chi-squared scores show the
terms most associated with the AI class.

### 1.3 Baseline
Multinomial Naive Bayes on TF-IDF, stratified 80/20 split (900 train, 225 test).

| Metric | Value |
|---|---|
| Train accuracy | 0.858 |
| Test accuracy | 0.813 (always predicting "classical" gives 0.791) |
| AI class: precision / recall / F1 | 1.00 / 0.11 / 0.19 |

Comment: accuracy is misleading because of class imbalance; the model finds only about 1 in 10 AI abstracts. Train and test
accuracy are close and the learning curve is low on both, so the baseline **under-fits** (high bias) rather than over-fits.

---

## Question 2: Models, enhancement and deployment (`notebooks/Q2_Models_Tuning_Deployment.ipynb`)

### 2.1 Four models versus the baseline
Same features and split. F1 of the AI class (test, and 5-fold cross-validated on the training data):

| Model | Test acc | Test F1 (AI) | CV F1 (AI) |
|---|---|---|---|
| Baseline: Multinomial NB | 0.813 | 0.19 | 0.18 |
| Logistic Regression | 0.858 | 0.67 | 0.75 |
| Linear SVM | 0.893 | 0.71 | 0.82 |
| Random Forest | 0.893 | 0.73 | 0.79 |
| Complement NB | 0.889 | 0.67 | 0.66 |

All four beat the baseline. Linear SVM and Random Forest are close; Linear SVM and Random Forest fit the training data almost
perfectly (some over-fitting). The test set has only 47 AI abstracts, so small differences are not reliable.

### 2.2 Enhancement: hyper-parameter tuning
Grid search with 5-fold cross-validation over `C`, the n-gram range and `min_df` for Logistic Regression and Linear SVM.
Logistic Regression improves from CV F1 0.75 to 0.82 (test F1 0.67 to 0.71); Linear SVM stays about 0.82.

### 2.3 Deployment
The tuned Logistic Regression (`C=30`, unigram TF-IDF, `min_df=2`, balanced class weights; CV F1 0.816, practically tied with the
tuned SVM at 0.815) is deployed as a Streamlit app (`deployment/app.py`). Final test result: accuracy 0.889, AI-class precision
0.79, recall 0.64, F1 0.71. See `deployment/README.md` for the steps.

> Add your screenshot of the running app here.

---

## Question 3: Topic analysis - research document mining (`notebooks/Q3_Document_Mining_Summarisation.ipynb`)

**3.1 Data frame:** the 5 articles in `research_corpus/` were read into a data frame (`article`, `content`), shape 5 x 2.

**3.2 Preprocessing:** whitespace normalised; URLs, e-mails, in-text citations (`(Smith, 2015)`, `[12]`), numbers and
punctuation removed; lowercase; NLTK stop words removed; WordNet lemmatisation. Articles shrink from 2,400 to 5,900 words
to 1,200 to 3,200 tokens.

**3.3 Summary of article 1 (frequency-based extractive):** sentences are scored by the normalised frequency of the words they
contain (divided by the square root of length) and the top 5 are kept in order. Article 1 goes from 5,602 words to about 180
words. Visualisations: top words, sentence scores with the selected sentences highlighted, and a word cloud.

**3.4 Aggregate summary of articles 2 to 5 (TextRank):** all sentences from the four articles are pooled, linked by TF-IDF cosine
similarity, and ranked with PageRank; the best 2 sentences per article form the aggregate summary. Visualisations: TextRank scores
by article, a word cloud of the aggregate corpus, and an article-to-article similarity heatmap, plus top TF-IDF terms per article.

---

## Question 4: LDA topic modelling (`notebooks/Q4_LDA_Topic_Modelling.ipynb`)

**4.1 LDA:** the articles are split into chunks of about 200 words (108 chunks, 908-term vocabulary) because five documents are
too few for LDA. Perplexity for 2 to 8 topics was compared as a guide and **4 topics** were chosen (the 5-topic model had two
overlapping topics). The notebook shows each topic's top 10 words with probabilities, bar charts of top words, the topic mix of
each article, and overall topic prevalence.

| Topic | Top words | Interpretation |
|---|---|---|
| 0 | data, triangulation, methods, researchers, qualitative | Triangulation and data in qualitative research |
| 1 | qualitative, mixed, reflexivity, researcher, quantitative | Mixed methods and researcher reflexivity |
| 2 | development, developmental, design, instructional, evaluation | Developmental and instructional design research |
| 3 | alpha, cronbach, instrument, reliability, scales | Cronbach's alpha and reliability |

research_5, research_3 and research_4 are each about 97% one topic. research_1 and research_2 mix topics 0 and 1.

**4.2 pyLDAvis:** Intertopic Distance Map built with `pyLDAvis.lda_model`; bubbles are topics (size = prevalence), selecting one shows
its most salient words, and the lambda slider balances frequency against topic-specificity. Saved as `docs/lda_intertopic_map.html`.

**4.3 Repository:** this repository.

---

## Limitations
The classifier's test set has only 47 AI abstracts, so metrics vary by a few points with a different split. The topic corpus is
5 articles about research methodology (not AI), so topics reflect methodology themes. LDA results depend on preprocessing and
the number of topics.
