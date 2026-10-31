# Computational Linguistics-2 — Assignment 3

## Movie Review Sentiment Classification

This submission implements the movie-review sentiment classification assignment using the Cornell **Polarity Dataset v2.0** (`txt_sentoken`).

The implementation is written from scratch and includes:

- Multinomial-style Naive Bayes with Laplace/add-one smoothing
- Binary Logistic Regression
- Cross-entropy loss
- Stochastic Gradient Descent (SGD)
- Seven required feature representations
- 10-fold evaluation using the predefined Cornell folds
- Precision, Recall, Accuracy, and F1-score
- Confusion matrices
- Prediction CSVs and report-ready result summaries
- Q4 error-analysis outputs
- Extra-credit Hu & Liu (2004) Opinion Lexicon features

---

## Directory Structure

The expected project structure is:

```text
Assignment3/
├── report/
│   └── report.pdf
│
└── src/
    ├── assignment3.py
    ├── assignment3_bonus.py
    │
    ├── hu_and_lin/
    │   ├── positive-words.txt
    │   └── negative-words.txt
    │
    ├── review_polarity/
    │   └── txt_sentoken/
    │       ├── pos/
    │       ├── neg/
    │       └── poldata.README.2.0
    │
    └── results/
```

The `hu_and_lin` directory contains the Hu & Liu (2004) positive and negative opinion-word lists.

---

## Requirements

The implementation is intentionally written without NumPy, scikit-learn, or other machine-learning libraries.

You need:

- Python 3
- The Cornell Polarity Dataset v2.0
- The Hu & Liu opinion lexicon for the extra-credit experiment

A virtual environment may be used, but it is not required.

---

# Part 1 — Main Assignment

The main program is:

```text
assignment3.py
```

It implements the seven feature representations required by the assignment.

### Feature 1 — BOW frequency

Bag-of-Words word-frequency features using a vocabulary constructed from the training data only.

### Feature 2 — BOW frequency with unseen test words

Bag-of-Words word-frequency features using the union of training and test vocabularies. Test labels and test word frequencies are not used to train the classifiers.

### Feature 3 — BOW binary

Binary Bag-of-Words representation. A word receives value 1 if it occurs at least once in the review.

### Feature 4 — Content-word frequency

Function words are removed using the function-word list specified in the assignment. The remaining content words are represented by their frequencies.

### Feature 5 — Content-word binary

Function words are removed and the remaining content words are represented using binary presence features.

### Feature 6 — BOW frequency + negation

The negation transformation is applied before constructing frequency features.

Negation markers include:

```text
not
no
never
words ending in n't
```

Words after a negation marker are prefixed with `NOT_` until the next punctuation mark.

Example:

```text
didn't like this movie, but I enjoyed it
```

becomes:

```text
didn't NOT_like NOT_this NOT_movie, but I enjoyed it
```

### Feature 7 — BOW binary + negation

The same negation transformation as Feature 6 is used, followed by binary Bag-of-Words features.

---

# Running the Main Program

All commands below should be run from the `src` directory.

First enter the project:

```bash
cd Assignment3/src
```

## Run all seven features

```bash
python assignment3.py --data review_polarity/txt_sentoken
```

This performs the full 10-fold evaluation.

## Run one feature

For example, Feature 1:

```bash
python assignment3.py \
  --data review_polarity/txt_sentoken \
  --feature 1
```

Replace `1` with any value from `1` to `7`.

## Run one feature on one fold

For example, Feature 1 on Fold 1:

```bash
python assignment3.py \
  --data review_polarity/txt_sentoken \
  --feature 1 \
  --fold 1
```

## Verbose Logistic Regression output

```bash
python assignment3.py \
  --data review_polarity/txt_sentoken \
  --feature 1 \
  --fold 1 \
  --verbose
```

## Optional Logistic Regression tuning

The implementation also supports an inner development split for tuning Logistic Regression:

```bash
python assignment3.py \
  --data review_polarity/txt_sentoken \
  --tune-lr
```

---

# Main Assignment Outputs

The program writes report-ready CSV files to:

```text
src/results/
```

Important outputs include:

```text
feature_1_fold_results.csv
feature_2_fold_results.csv
...
feature_7_fold_results.csv

feature_1_predictions.csv
feature_2_predictions.csv
...
feature_7_predictions.csv

all_features_summary.csv
```

The summary file contains the final performance of Naive Bayes and Logistic Regression for all seven feature representations.

The prediction files are also used for the Q4 error analysis.

---

# Part 2 — Extra Credit

The extra-credit experiment uses the:

**Hu & Liu (2004) Opinion Lexicon**

The lexicon contains separate positive and negative word lists:

```text
hu_and_lin/
├── positive-words.txt
└── negative-words.txt
```

The bonus experiment augments **Feature V (content-word binary features)** with two lexicon-based features:

```text
POS_LEXICON_COUNT
NEG_LEXICON_COUNT
```

`POS_LEXICON_COUNT` is the number of positive Hu & Liu words occurring in the review.

`NEG_LEXICON_COUNT` is the number of negative Hu & Liu words occurring in the review.

The lexicon is treated as a fixed external resource and is not learned from the movie-review dataset.

---

# Running the Extra-Credit Experiment

Run the following command from:

```text
Assignment3/src
```

```bash
python assignment3_bonus.py \
  --data review_polarity/txt_sentoken \
  --bonus-lexicon \
  --lexicon-dir hu_and_lin
```

The program performs the Hu & Liu experiment over all 10 predefined folds.

The output is written to:

```text
src/results/
```

Important bonus output files include:

```text
bonus_hu_liu_fold_results.csv
bonus_hu_liu_predictions.csv
```

The terminal output also reports the pooled 10-fold results and macro-averaged metrics.

---

# Extra-Credit Results

For the experiment reported in this submission, the pooled 10-fold results were:

| Classifier | Precision | Recall | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Naive Bayes + Hu & Liu | 0.8489 | 0.8200 | 0.8370 | 0.8342 |
| Logistic Regression + Hu & Liu | 0.8332 | 0.8640 | 0.8455 | 0.8483 |

The original Feature V results were:

| Classifier | Original Feature V F1 | Hu & Liu F1 | Change |
|---|---:|---:|---:|
| Naive Bayes | 0.8256 | 0.8342 | +0.0086 |
| Logistic Regression | 0.8670 | 0.8483 | -0.0187 |

The lexicon therefore improved Naive Bayes but decreased the F1-score of Logistic Regression. The experiment is reported as observed; no attempt was made to alter the results to force an improvement.

---

# Evaluation Protocol

The Cornell dataset contains 2000 reviews:

- 1000 positive
- 1000 negative

The predefined folds are used as follows:

```text
Fold 1: 200 test reviews, 1800 training reviews
Fold 2: 200 test reviews, 1800 training reviews
...
Fold 10: 200 test reviews, 1800 training reviews
```

Every review is used exactly once as a test example.

Each fold is trained and evaluated independently from scratch.

The reported final metrics are based on the ten test folds.

---

# Assumptions and Preprocessing

The following assumptions are used:

1. Review text is converted to lowercase before tokenization.
2. The Cornell files are read using `latin-1` encoding because the original corpus may contain non-UTF-8 bytes.
3. No stemming or lemmatization is performed.
4. Lexicon matching is based on exact lowercase word matches.
5. Punctuation is not counted as a lexicon entry.
6. Multiple occurrences of a lexicon word contribute multiple times to the lexicon count.
7. The Hu & Liu lexicon is treated as a fixed external resource.
8. The same predefined 10-fold evaluation is used for the extra-credit experiment.
9. The basic Hu & Liu count features are direct word matches and are not independently negation-aware.
10. For Feature 2, the union of training and test vocabularies is intentionally used because this is explicitly required by the assignment. Test labels and test word frequencies are not used to estimate classifier parameters.

---

# Reproducibility

The main experiments use a fixed random seed and deterministic fold assignment based on the Cornell filenames.

The default Logistic Regression settings in the program are:

```text
Learning rate: 0.001
Epochs: 20
Random seed: 42
```

These can be changed through command-line arguments when required.

For example:

```bash
python assignment3_bonus.py \
  --data review_polarity/txt_sentoken \
  --bonus-lexicon \
  --lexicon-dir hu_and_lin \
  --lr 0.001 \
  --epochs 20 \
  --seed 42
```

---

# Files Submitted

The important source files are:

```text
assignment3.py
```

Main implementation for Questions 1–4.

```text
assignment3_bonus.py
```

Implementation containing the extra-credit Hu & Liu experiment.

```text
hu_and_lin/
```

Contains the Hu & Liu positive and negative opinion lexicons.

```text
review_polarity/
```

Contains the Cornell movie-review dataset.

```text
report/report.pdf
```

Contains the written report and experimental results.

