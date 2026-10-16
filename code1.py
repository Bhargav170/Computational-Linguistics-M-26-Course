import csv
import json
import math
import unicodedata
from collections import Counter


TRAIN_FILE = "./Assignment2/Language_datasets/polish_czech_train.csv"
TEST_FILE = "./Assignment2/Language_datasets/polish_czech_test.csv"


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(sentence):
    """
    Preprocessing assumptions:
    - UTF-8 encoding
    - Unicode NFC normalization
    - lowercase
    - leading/trailing whitespace removed
    - spaces retained
    - punctuation retained
    - sentence boundary markers added
    """

    sentence = unicodedata.normalize("NFC", sentence)
    sentence = sentence.strip().lower()

    return "^" + sentence + "$"


# ============================================================
# N-GRAM FUNCTIONS
# ============================================================

def get_ngrams(text, n):
    """
    Return character n-gram counts.
    """

    return Counter(
        text[i:i + n]
        for i in range(len(text) - n + 1)
    )


def build_ngram_counts(sentences):
    """
    Build unigram, bigram, trigram and 4-gram counts.
    """

    counts = {
        1: Counter(),
        2: Counter(),
        3: Counter(),
        4: Counter()
    }

    for sentence in sentences:

        sentence = preprocess(sentence)

        for n in range(1, 5):
            counts[n].update(
                get_ngrams(sentence, n)
            )

    return counts


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(filename):

    polish = []
    czech = []

    with open(
        filename,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            language = row["lan_code"]
            sentence = row["sentence"]

            if language == "pol":
                polish.append(sentence)

            elif language == "ces":
                czech.append(sentence)

    return polish, czech


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data(filename):

    test_data = []

    with open(
        filename,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            test_data.append(
                (
                    row["lan_code"],
                    row["sentence"]
                )
            )

    return test_data


# ============================================================
# SAVE N-GRAM COUNTS
# ============================================================

def save_counts(counts, language):

    for n in range(1, 5):

        filename = f"{language}_{n}gram_counts.json"

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                counts[n],
                f,
                ensure_ascii=False,
                indent=2
            )


# ============================================================
# PRINT N-GRAM SUMMARY
# ============================================================

def print_summary(counts, language):

    print(f"\n===== {language.upper()} =====")

    for n in range(1, 5):

        counter = counts[n]

        print(f"\n{n}-gram")
        print("-" * 30)

        print(
            "Total occurrences:",
            sum(counter.values())
        )

        print(
            "Distinct n-grams:",
            len(counter)
        )

        print("Top 10:")

        for gram, count in counter.most_common(10):

            print(
                repr(gram),
                ":",
                count
            )


# ============================================================
# LOG PROBABILITY
# ============================================================

def calculate_log_probability(
    sentence,
    n,
    language_counts,
    vocabulary
):
    """
    Calculate log probability of a sentence
    using an n-gram language model with
    Laplace (add-one) smoothing.
    """

    sentence_ngrams = get_ngrams(
        sentence,
        n
    )

    total_ngrams = sum(
        language_counts.values()
    )

    vocabulary_size = len(vocabulary)

    log_probability = 0.0

    for ngram, test_count in sentence_ngrams.items():

        # IMPORTANT:
        # Use the TRAINING count of the n-gram.
        training_count = language_counts[ngram]

        probability = (
            training_count + 1
        ) / (
            total_ngrams + vocabulary_size
        )

        log_probability += (
            test_count * math.log(probability)
        )

    return log_probability


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_sentence(
    sentence,
    n,
    polish_counts,
    czech_counts,
    vocabulary
):
    """
    Classify a sentence as Polish or Czech.
    """

    polish_score = calculate_log_probability(
        sentence,
        n,
        polish_counts,
        vocabulary
    )

    czech_score = calculate_log_probability(
        sentence,
        n,
        czech_counts,
        vocabulary
    )

    if polish_score > czech_score:
        return "pol"

    return "ces"


# ============================================================
# PRECISION AND RECALL
# ============================================================

def calculate_metrics(results):

    tp_pol = fp_pol = fn_pol = 0
    tp_ces = 0

    for actual, predicted in results:

        if actual == "pol" and predicted == "pol":
            tp_pol += 1

        elif actual == "ces" and predicted == "pol":
            fp_pol += 1

        elif actual == "pol" and predicted == "ces":
            fn_pol += 1

        elif actual == "ces" and predicted == "ces":
            tp_ces += 1

    # Binary classification:
    #
    # Polish FN = Czech FP
    # Polish FP = Czech FN

    fp_ces = fn_pol
    fn_ces = fp_pol

    precision_pol = (
        tp_pol / (tp_pol + fp_pol)
        if tp_pol + fp_pol
        else 0
    )

    recall_pol = (
        tp_pol / (tp_pol + fn_pol)
        if tp_pol + fn_pol
        else 0
    )

    precision_ces = (
        tp_ces / (tp_ces + fp_ces)
        if tp_ces + fp_ces
        else 0
    )

    recall_ces = (
        tp_ces / (tp_ces + fn_ces)
        if tp_ces + fn_ces
        else 0
    )

    return (
        precision_pol,
        recall_pol,
        precision_ces,
        recall_ces
    )


# ============================================================
# ZERO-COUNT N-GRAMS
# ============================================================

def count_zero_ngrams(
    test_sentences,
    n,
    language_counts
):
    """
    Count n-gram occurrences in the test set
    that were unseen in the corresponding
    training language.
    """

    zero_occurrences = 0
    zero_distinct = set()

    for sentence in test_sentences:

        sentence = preprocess(sentence)

        sentence_ngrams = get_ngrams(
            sentence,
            n
        )

        for ngram, count in sentence_ngrams.items():

            if language_counts[ngram] == 0:

                zero_occurrences += count
                zero_distinct.add(ngram)

    return (
        zero_occurrences,
        len(zero_distinct)
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

# ------------------------------------------------------------
# Q1: LOAD TRAINING DATA
# ------------------------------------------------------------

polish_sentences, czech_sentences = load_training_data(
    TRAIN_FILE
)

print(
    "Number of Polish training sentences:",
    len(polish_sentences)
)

print(
    "Number of Czech training sentences:",
    len(czech_sentences)
)


# ------------------------------------------------------------
# Q1: BUILD N-GRAM COUNTS
# ------------------------------------------------------------

polish_counts = build_ngram_counts(
    polish_sentences
)

czech_counts = build_ngram_counts(
    czech_sentences
)


# ------------------------------------------------------------
# Q1: DISPLAY COUNTS
# ------------------------------------------------------------

print_summary(
    polish_counts,
    "polish"
)

print_summary(
    czech_counts,
    "czech"
)


# ------------------------------------------------------------
# Q1: SAVE COUNTS
# ------------------------------------------------------------

save_counts(
    polish_counts,
    "polish"
)

save_counts(
    czech_counts,
    "czech"
)


# ------------------------------------------------------------
# Q2: LOAD TEST DATA
# ------------------------------------------------------------

test_data = load_test_data(
    TEST_FILE
)

print(
    "\nNumber of test sentences:",
    len(test_data)
)


# ------------------------------------------------------------
# SEPARATE TEST SENTENCES BY LANGUAGE
# ------------------------------------------------------------

polish_test = [
    sentence
    for language, sentence in test_data
    if language == "pol"
]

czech_test = [
    sentence
    for language, sentence in test_data
    if language == "ces"
]


# ------------------------------------------------------------
# Q2: CLASSIFICATION
# ------------------------------------------------------------

all_results = {}

for n in range(1, 5):

    # Vocabulary shared by both language models
    vocabulary = (
        set(polish_counts[n].keys())
        |
        set(czech_counts[n].keys())
    )

    results = []

    for actual, sentence in test_data:

        processed_sentence = preprocess(
            sentence
        )

        predicted = classify_sentence(
            processed_sentence,
            n,
            polish_counts[n],
            czech_counts[n],
            vocabulary
        )

        results.append(
            (actual, predicted)
        )

    all_results[n] = results


# ------------------------------------------------------------
# Q2: PRECISION AND RECALL
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("PRECISION AND RECALL")
print("=" * 75)

print(
    f"{'N-gram':<10}"
    f"{'Pol Precision':<18}"
    f"{'Pol Recall':<15}"
    f"{'Czech Precision':<18}"
    f"{'Czech Recall':<15}"
)

for n in range(1, 5):

    (
        precision_pol,
        recall_pol,
        precision_ces,
        recall_ces
    ) = calculate_metrics(
        all_results[n]
    )

    print(
        f"{n}-gram    "
        f"{precision_pol:.4f}           "
        f"{recall_pol:.4f}        "
        f"{precision_ces:.4f}           "
        f"{recall_ces:.4f}"
    )


# ------------------------------------------------------------
# Q2: ZERO-COUNT N-GRAMS
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("ZERO-COUNT N-GRAMS")
print("=" * 75)

for n in range(1, 5):

    (
        pol_zero_occurrences,
        pol_zero_distinct
    ) = count_zero_ngrams(
        polish_test,
        n,
        polish_counts[n]
    )

    (
        ces_zero_occurrences,
        ces_zero_distinct
    ) = count_zero_ngrams(
        czech_test,
        n,
        czech_counts[n]
    )

    print(f"\n{n}-gram:")

    print(
        "Polish test n-grams unseen in Polish training:"
    )

    print(
        "  Occurrences:",
        pol_zero_occurrences
    )

    print(
        "  Distinct n-grams:",
        pol_zero_distinct
    )

    print(
        "Czech test n-grams unseen in Czech training:"
    )

    print(
        "  Occurrences:",
        ces_zero_occurrences
    )

    print(
        "  Distinct n-grams:",
        ces_zero_distinct
    )