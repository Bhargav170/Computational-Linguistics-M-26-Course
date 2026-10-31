import csv
import json
import math
import unicodedata
from collections import Counter


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = "./Assignment2/Language_datasets/spanish_portugese_train.csv"
TEST_FILE = "./Assignment2/Language_datasets/spanish_portugese_test.csv"


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

    language_data = {}

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

            if language not in language_data:
                language_data[language] = []

            language_data[language].append(sentence)

    return language_data


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

        # Training count of this n-gram
        training_count = language_counts[ngram]

        # Laplace / add-one smoothing
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
    language1_counts,
    language2_counts,
    vocabulary,
    language1,
    language2
):

    language1_score = calculate_log_probability(
        sentence,
        n,
        language1_counts,
        vocabulary
    )

    language2_score = calculate_log_probability(
        sentence,
        n,
        language2_counts,
        vocabulary
    )

    if language1_score > language2_score:
        return language1

    return language2


# ============================================================
# PRECISION AND RECALL
# ============================================================

def calculate_metrics(
    results,
    language1,
    language2
):

    tp_l1 = fp_l1 = fn_l1 = 0
    tp_l2 = 0

    for actual, predicted in results:

        if actual == language1 and predicted == language1:

            tp_l1 += 1

        elif actual == language2 and predicted == language1:

            fp_l1 += 1

        elif actual == language1 and predicted == language2:

            fn_l1 += 1

        elif actual == language2 and predicted == language2:

            tp_l2 += 1

    # Binary classification
    fp_l2 = fn_l1
    fn_l2 = fp_l1

    precision_l1 = (
        tp_l1 / (tp_l1 + fp_l1)
        if tp_l1 + fp_l1
        else 0
    )

    recall_l1 = (
        tp_l1 / (tp_l1 + fn_l1)
        if tp_l1 + fn_l1
        else 0
    )

    precision_l2 = (
        tp_l2 / (tp_l2 + fp_l2)
        if tp_l2 + fp_l2
        else 0
    )

    recall_l2 = (
        tp_l2 / (tp_l2 + fn_l2)
        if tp_l2 + fn_l2
        else 0
    )

    return (
        precision_l1,
        recall_l1,
        precision_l2,
        recall_l2
    )


# ============================================================
# ZERO-COUNT N-GRAMS
# ============================================================

def count_zero_ngrams(
    test_sentences,
    n,
    language_counts
):

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

language_data = load_training_data(
    TRAIN_FILE
)

# Make sure exactly two languages are present
languages = list(language_data.keys())

if len(languages) != 2:

    raise ValueError(
        f"Expected exactly 2 languages, "
        f"but found: {languages}"
    )

language1 = languages[0]
language2 = languages[1]

language1_sentences = language_data[language1]
language2_sentences = language_data[language2]


print(
    f"Number of {language1} training sentences:",
    len(language1_sentences)
)

print(
    f"Number of {language2} training sentences:",
    len(language2_sentences)
)


# ------------------------------------------------------------
# Q1: BUILD N-GRAM COUNTS
# ------------------------------------------------------------

language1_counts = build_ngram_counts(
    language1_sentences
)

language2_counts = build_ngram_counts(
    language2_sentences
)


# ------------------------------------------------------------
# Q1: DISPLAY COUNTS
# ------------------------------------------------------------

print_summary(
    language1_counts,
    language1
)

print_summary(
    language2_counts,
    language2
)


# ------------------------------------------------------------
# Q1: SAVE COUNTS
# ------------------------------------------------------------

save_counts(
    language1_counts,
    language1
)

save_counts(
    language2_counts,
    language2
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

language1_test = [
    sentence
    for language, sentence in test_data
    if language == language1
]

language2_test = [
    sentence
    for language, sentence in test_data
    if language == language2
]


# ------------------------------------------------------------
# Q2: CLASSIFICATION
# ------------------------------------------------------------

all_results = {}

for n in range(1, 5):

    # Vocabulary shared by both language models
    vocabulary = (
        set(language1_counts[n].keys())
        |
        set(language2_counts[n].keys())
    )

    results = []

    for actual, sentence in test_data:

        processed_sentence = preprocess(
            sentence
        )

        predicted = classify_sentence(
            processed_sentence,
            n,
            language1_counts[n],
            language2_counts[n],
            vocabulary,
            language1,
            language2
        )

        results.append(
            (actual, predicted)
        )

    all_results[n] = results


# ------------------------------------------------------------
# Q2: PRECISION AND RECALL
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("PRECISION AND RECALL")
print("=" * 80)

print(
    f"{'N-gram':<10}"
    f"{language1 + ' Precision':<20}"
    f"{language1 + ' Recall':<18}"
    f"{language2 + ' Precision':<20}"
    f"{language2 + ' Recall':<18}"
)


for n in range(1, 5):

    (
        precision_l1,
        recall_l1,
        precision_l2,
        recall_l2
    ) = calculate_metrics(
        all_results[n],
        language1,
        language2
    )

    print(
        f"{n}-gram    "
        f"{precision_l1:.4f}              "
        f"{recall_l1:.4f}          "
        f"{precision_l2:.4f}              "
        f"{recall_l2:.4f}"
    )


# ------------------------------------------------------------
# Q2: ZERO-COUNT N-GRAMS
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("ZERO-COUNT N-GRAMS")
print("=" * 80)


for n in range(1, 5):

    (
        l1_zero_occurrences,
        l1_zero_distinct
    ) = count_zero_ngrams(
        language1_test,
        n,
        language1_counts[n]
    )

    (
        l2_zero_occurrences,
        l2_zero_distinct
    ) = count_zero_ngrams(
        language2_test,
        n,
        language2_counts[n]
    )

    print(f"\n{n}-gram:")

    print(
        f"{language1} test n-grams unseen "
        f"in {language1} training:"
    )

    print(
        "  Occurrences:",
        l1_zero_occurrences
    )

    print(
        "  Distinct n-grams:",
        l1_zero_distinct
    )

    print(
        f"{language2} test n-grams unseen "
        f"in {language2} training:"
    )

    print(
        "  Occurrences:",
        l2_zero_occurrences
    )

    print(
        "  Distinct n-grams:",
        l2_zero_distinct
    )