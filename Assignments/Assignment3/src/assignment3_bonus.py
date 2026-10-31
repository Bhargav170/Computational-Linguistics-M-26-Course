#!/usr/bin/env python3
"""
Assignment 3 - Movie Review Sentiment Classification

Scratch implementations:
  1. Naive Bayes with Laplace/add-one smoothing
  2. Binary Logistic Regression with cross-entropy + SGD

Seven assignment feature representations:
  1. BOW word frequencies, vocabulary from training only
  2. BOW word frequencies, vocabulary = train U test
  3. BOW binary/presence features
  4. Content-word frequencies, function words removed
  5. Content-word binary/presence features
  6. BOW frequencies after negation transformation
  7. BOW binary after negation transformation

Also included:
  - 10-fold evaluation using the Cornell cv000..cv999 fold IDs
  - optional inner development split for LR hyperparameter tuning
  - confusion matrices
  - precision, recall, accuracy, F1
  - prediction CSVs
  - Q4 error-analysis CSV
  - LR learned-weight inspection

Important:
  * This is a from-scratch implementation. NumPy/sklearn are NOT required.
  * Extra-credit Hu & Liu polarity-lexicon features can be run with
    --bonus-lexicon. They are added to Feature V (content-word binary).
  * For content-word features, the exact English function/stop-word list
    printed at the end of Assignment3.pdf is used below.
  * Q1(ii) intentionally exposes TEST vocabulary to the feature dictionary
    because the assignment explicitly asks for "including unseen words in
    the test set". Test counts and labels are never used for learning.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Labels and feature names
# ---------------------------------------------------------------------------

NEGATIVE = 0
POSITIVE = 1

FEATURE_NAMES = {
    1: "BOW frequency (train vocabulary)",
    2: "BOW frequency (train + test vocabulary)",
    3: "BOW binary",
    4: "Content-word frequency",
    5: "Content-word binary",
    6: "BOW frequency + negation",
    7: "BOW binary + negation",
}


# ---------------------------------------------------------------------------
# Exact English function/stop-word list printed in Assignment3.pdf
# (courtesy of the NLTK toolkit).
# ---------------------------------------------------------------------------

FUNCTION_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "you're", "you've", "you'll", "you'd", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "that'll", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "don't", "should", "should've", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
    "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan",
    "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't"
}

# ---------------------------------------------------------------------------
# Data representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Review:
    filename: str
    text: str
    label: int
    fold: int


@dataclass
class Metrics:
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def accuracy(self) -> float:
        d = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r) if (p + r) else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_fold_from_filename(filename: str) -> int:
    """
    Cornell polarity v2.0:
      cv000..cv099 -> fold 1
      cv100..cv199 -> fold 2
      ...
      cv900..cv999 -> fold 10
    """
    m = re.match(r"cv(\d{3})_", filename)
    if not m:
        raise ValueError(f"Unexpected corpus filename: {filename}")
    n = int(m.group(1))
    return n // 100 + 1


def load_reviews(data_dir: str | Path) -> List[Review]:
    data_dir = Path(data_dir)
    pos_dir = data_dir / "pos"
    neg_dir = data_dir / "neg"

    if not pos_dir.is_dir() or not neg_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {pos_dir} and {neg_dir}. "
            "Pass --data pointing to txt_sentoken."
        )

    reviews: List[Review] = []

    for directory, label in [(neg_dir, NEGATIVE), (pos_dir, POSITIVE)]:
        for path in sorted(directory.glob("*.txt")):
            # The Cornell corpus can contain non-UTF-8 bytes.
            text = path.read_text(encoding="latin-1")
            reviews.append(
                Review(
                    filename=path.name,
                    text=text,
                    label=label,
                    fold=get_fold_from_filename(path.name),
                )
            )

    if len(reviews) != 2000:
        print(
            f"WARNING: expected 2000 reviews for polarity v2.0, "
            f"found {len(reviews)}."
        )

    return sorted(reviews, key=lambda r: (r.fold, r.label, r.filename))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Baseline word tokenizer.
# Lowercase is applied before tokenization.
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Negation tokenizer keeps punctuation as separate tokens so that punctuation
# can terminate a negation scope.
NEGATION_PATTERN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|[.!?,;:()\[\]{}\"']"
)


def _normalize_text(text: str) -> str:
    # The Cornell corpus is normally ASCII, but normalize common curly
    # apostrophes so contractions such as “didn’t” are handled consistently.
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")


def tokenize(text: str) -> List[str]:
    return WORD_PATTERN.findall(_normalize_text(text))


def tokenize_with_punctuation(text: str) -> List[str]:
    return NEGATION_PATTERN.findall(_normalize_text(text))


def is_punctuation(token: str) -> bool:
    return bool(re.fullmatch(r"[.!?,;:()\[\]{}\"']", token))


def is_negation_trigger(token: str) -> bool:
    """
    Assignment triggers:
      n’t / n't
      not
      no
      never

    The corpus may contain "didn't" rather than the separated Penn
    Treebank-style "did n't", so both forms are supported.
    """
    return token in {"not", "no", "never"} or token.endswith("n't")


def apply_negation_feature(text: str) -> List[str]:
    """
    Prefix NOT_ to every word after a negation trigger until the next
    punctuation mark.

    Example:
      didn't like this movie, but I
    becomes approximately:
      didn't NOT_like NOT_this NOT_movie , but I

    The trigger itself is retained unchanged.
    """
    tokens = tokenize_with_punctuation(text)
    output: List[str] = []
    negating = False

    for token in tokens:
        if is_punctuation(token):
            output.append(token)
            negating = False
            continue

        if is_negation_trigger(token):
            output.append(token)
            negating = True
            continue

        if negating:
            output.append("NOT_" + token)
        else:
            output.append(token)

    # Feature extraction later ignores punctuation.
    return output


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def content_tokens(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if t not in FUNCTION_WORDS]


def prepare_tokens(review: Review, feature: int) -> List[str]:
    if feature in {6, 7}:
        tokens = apply_negation_feature(review.text)
        # Ignore punctuation after negation scope has been established.
        return [t for t in tokens if not is_punctuation(t)]

    tokens = tokenize(review.text)

    if feature in {4, 5}:
        tokens = content_tokens(tokens)

    return tokens


def build_vocabulary(
    documents: Sequence[Review],
    feature: int,
    include_test: Sequence[Review] | None = None,
) -> Dict[str, int]:
    """
    Q1(ii) intentionally uses train + test vocabulary.
    All other representations use training vocabulary only.
    """
    token_set = set()

    for review in documents:
        token_set.update(prepare_tokens(review, feature))

    if feature == 2 and include_test is not None:
        for review in include_test:
            token_set.update(prepare_tokens(review, feature))

    return {word: i for i, word in enumerate(sorted(token_set))}


def vectorize(
    review: Review,
    feature: int,
    vocabulary: Dict[str, int],
) -> Dict[int, float]:
    """
    Sparse vector:
      key   = feature index
      value = count or 1

    Features 3, 5, 7 are binary.
    """
    tokens = prepare_tokens(review, feature)

    counts = Counter(
        token for token in tokens
        if token in vocabulary
    )

    binary = feature in {3, 5, 7}

    if binary:
        return {
            vocabulary[token]: 1.0
            for token in counts
        }

    return {
        vocabulary[token]: float(count)
        for token, count in counts.items()
    }


def make_vectors(
    reviews: Sequence[Review],
    feature: int,
    vocabulary: Dict[str, int],
) -> List[Dict[int, float]]:
    return [
        vectorize(review, feature, vocabulary)
        for review in reviews
    ]


# ---------------------------------------------------------------------------
# Extra credit: Hu & Liu (2004) Opinion Lexicon
# ---------------------------------------------------------------------------

def load_hu_liu_lexicon(lexicon_dir: str | Path) -> Tuple[set[str], set[str]]:
    """Load the original Hu & Liu positive/negative word lists.

    Lines beginning with ';' are comments in the distributed lexicon files.
    Matching is exact after lowercasing, with no stemming or lemmatization.
    """
    lexicon_dir = Path(lexicon_dir)
    pos_path = lexicon_dir / "positive-words.txt"
    neg_path = lexicon_dir / "negative-words.txt"

    if not pos_path.is_file() or not neg_path.is_file():
        raise FileNotFoundError(
            f"Expected {pos_path} and {neg_path}. "
            "Pass --lexicon-dir pointing to the hu_and_lin folder."
        )

    def read_words(path: Path) -> set[str]:
        words: set[str] = set()
        with path.open("r", encoding="latin-1") as f:
            for line in f:
                word = line.strip().lower()
                if not word or word.startswith(";"):
                    continue
                words.add(word)
        return words

    positive = read_words(pos_path)
    negative = read_words(neg_path)
    return positive, negative


def lexicon_counts(
    review: Review,
    positive_words: set[str],
    negative_words: set[str],
) -> Tuple[int, int]:
    """Return (positive_count, negative_count) for one review.

    Counts are based on the same tokenizer used by the assignment's ordinary
    BOW features. Negation is NOT applied here; this bonus first measures the
    direct Hu & Liu lexical evidence and adds it to Feature V.
    """
    tokens = tokenize(review.text)
    positive_count = sum(token in positive_words for token in tokens)
    negative_count = sum(token in negative_words for token in tokens)
    return positive_count, negative_count


def augment_with_lexicon(
    base_vectors: Sequence[Dict[int, float]],
    reviews: Sequence[Review],
    base_dimension: int,
    positive_words: set[str],
    negative_words: set[str],
) -> List[Dict[int, float]]:
    """Append two count features to Feature V:

      base_dimension     = POS_LEXICON_COUNT
      base_dimension + 1 = NEG_LEXICON_COUNT

    The ordinary Feature V binary content-word features are left unchanged.
    """
    augmented: List[Dict[int, float]] = []
    pos_index = base_dimension
    neg_index = base_dimension + 1

    for vector, review in zip(base_vectors, reviews):
        pos_count, neg_count = lexicon_counts(
            review, positive_words, negative_words
        )
        new_vector = dict(vector)
        if pos_count:
            new_vector[pos_index] = float(pos_count)
        if neg_count:
            new_vector[neg_index] = float(neg_count)
        augmented.append(new_vector)

    return augmented


def run_lexicon_bonus_fold(
    all_reviews: Sequence[Review],
    test_fold: int,
    positive_words: set[str],
    negative_words: set[str],
    lr: float,
    epochs: int,
    seed: int,
    verbose: bool = False,
) -> Dict:
    """Run Feature V + Hu & Liu lexicon features on one outer fold."""
    test_reviews = [r for r in all_reviews if r.fold == test_fold]
    train_reviews = [r for r in all_reviews if r.fold != test_fold]

    # Start from the strongest required representation: Feature V.
    vocabulary = build_vocabulary(train_reviews, feature=5)
    X_train_base = make_vectors(train_reviews, feature=5, vocabulary=vocabulary)
    X_test_base = make_vectors(test_reviews, feature=5, vocabulary=vocabulary)

    base_dimension = len(vocabulary)
    X_train = augment_with_lexicon(
        X_train_base, train_reviews, base_dimension,
        positive_words, negative_words,
    )
    X_test = augment_with_lexicon(
        X_test_base, test_reviews, base_dimension,
        positive_words, negative_words,
    )

    y_train = [r.label for r in train_reviews]
    y_test = [r.label for r in test_reviews]

    nb = NaiveBayes(base_dimension + 2)
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_metrics = evaluate(y_test, nb_pred)

    lr_model = LogisticRegressionSGD(
        base_dimension + 2,
        learning_rate=lr,
        epochs=epochs,
        seed=seed + test_fold,
    )
    lr_model.fit(X_train, y_train, verbose=verbose)
    lr_pred = lr_model.predict(X_test)
    lr_metrics = evaluate(y_test, lr_pred)

    return {
        "fold": test_fold,
        "train_size": len(train_reviews),
        "test_size": len(test_reviews),
        "vocabulary_size": len(vocabulary),
        "total_feature_dimension": base_dimension + 2,
        "nb_metrics": nb_metrics,
        "lr_metrics": lr_metrics,
        "test_reviews": test_reviews,
        "y_test": y_test,
        "nb_pred": nb_pred,
        "lr_pred": lr_pred,
    }


def run_lexicon_bonus_cv(
    reviews: Sequence[Review],
    lexicon_dir: str | Path,
    lr: float,
    epochs: int,
    seed: int,
    verbose: bool = False,
    output_dir: Path | None = None,
) -> Dict:
    """Run the Hu & Liu bonus experiment over all 10 predefined folds."""
    positive_words, negative_words = load_hu_liu_lexicon(lexicon_dir)

    print("\n" + "=" * 80)
    print("EXTRA CREDIT: FEATURE V + HU & LIU (2004) OPINION LEXICON")
    print("=" * 80)
    print(f"Positive lexicon entries: {len(positive_words)}")
    print(f"Negative lexicon entries: {len(negative_words)}")
    print("Added features: POS_LEXICON_COUNT, NEG_LEXICON_COUNT")

    aggregate_true: List[int] = []
    aggregate_nb: List[int] = []
    aggregate_lr: List[int] = []
    fold_rows: List[Dict] = []
    all_test_reviews: List[Review] = []

    for fold in range(1, 11):
        result = run_lexicon_bonus_fold(
            reviews, fold, positive_words, negative_words,
            lr=lr, epochs=epochs, seed=seed, verbose=verbose,
        )
        nb_m = result["nb_metrics"]
        lr_m = result["lr_metrics"]

        print(f"\nFold {fold}/10")
        print_metrics("  Naive Bayes", nb_m)
        print_metrics("  Logistic Regression", lr_m)

        aggregate_true.extend(result["y_test"])
        aggregate_nb.extend(result["nb_pred"])
        aggregate_lr.extend(result["lr_pred"])
        all_test_reviews.extend(result["test_reviews"])

        fold_rows.append({
            "fold": fold,
            "train_size": result["train_size"],
            "test_size": result["test_size"],
            "vocabulary_size": result["vocabulary_size"],
            "total_feature_dimension": result["total_feature_dimension"],
            "nb_precision": nb_m.precision,
            "nb_recall": nb_m.recall,
            "nb_accuracy": nb_m.accuracy,
            "nb_f1": nb_m.f1,
            "nb_tn": nb_m.tn,
            "nb_fp": nb_m.fp,
            "nb_fn": nb_m.fn,
            "nb_tp": nb_m.tp,
            "lr_precision": lr_m.precision,
            "lr_recall": lr_m.recall,
            "lr_accuracy": lr_m.accuracy,
            "lr_f1": lr_m.f1,
            "lr_tn": lr_m.tn,
            "lr_fp": lr_m.fp,
            "lr_fn": lr_m.fn,
            "lr_tp": lr_m.tp,
        })

    nb_all = evaluate(aggregate_true, aggregate_nb)
    lr_all = evaluate(aggregate_true, aggregate_lr)

    macro = {
        "nb_precision": mean(r["nb_precision"] for r in fold_rows),
        "nb_recall": mean(r["nb_recall"] for r in fold_rows),
        "nb_accuracy": mean(r["nb_accuracy"] for r in fold_rows),
        "nb_f1": mean(r["nb_f1"] for r in fold_rows),
        "lr_precision": mean(r["lr_precision"] for r in fold_rows),
        "lr_recall": mean(r["lr_recall"] for r in fold_rows),
        "lr_accuracy": mean(r["lr_accuracy"] for r in fold_rows),
        "lr_f1": mean(r["lr_f1"] for r in fold_rows),
    }

    print("\n" + "-" * 80)
    print("10-FOLD POOLED RESULTS: FEATURE V + HU & LIU")
    print("-" * 80)
    print_metrics("Naive Bayes", nb_all)
    print_metrics("Logistic Regression", lr_all)
    print("\n10-fold macro-average:")
    print(
        f"  NB: Precision={macro['nb_precision']:.4f}, "
        f"Recall={macro['nb_recall']:.4f}, "
        f"Accuracy={macro['nb_accuracy']:.4f}, F1={macro['nb_f1']:.4f}"
    )
    print(
        f"  LR: Precision={macro['lr_precision']:.4f}, "
        f"Recall={macro['lr_recall']:.4f}, "
        f"Accuracy={macro['lr_accuracy']:.4f}, F1={macro['lr_f1']:.4f}"
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_fold_rows(output_dir / "bonus_hu_liu_fold_results.csv", fold_rows)
        write_predictions(
            output_dir / "bonus_hu_liu_predictions.csv",
            all_test_reviews, aggregate_true, aggregate_nb, aggregate_lr,
        )
        with (output_dir / "bonus_hu_liu_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "experiment", "classifier", "pooled_precision", "pooled_recall",
                "pooled_accuracy", "pooled_f1", "tn", "fp", "fn", "tp",
                "macro_precision", "macro_recall", "macro_accuracy", "macro_f1"
            ])
            for name, m, prefix in [
                ("Naive Bayes", nb_all, "nb_"),
                ("Logistic Regression", lr_all, "lr_"),
            ]:
                writer.writerow([
                    "Feature V + Hu & Liu", name,
                    m.precision, m.recall, m.accuracy, m.f1,
                    m.tn, m.fp, m.fn, m.tp,
                    macro[prefix + "precision"], macro[prefix + "recall"],
                    macro[prefix + "accuracy"], macro[prefix + "f1"],
                ])

    return {
        "nb": nb_all,
        "lr": lr_all,
        "macro": macro,
        "fold_rows": fold_rows,
        "reviews": all_test_reviews,
        "true": aggregate_true,
        "nb_pred": aggregate_nb,
        "lr_pred": aggregate_lr,
        "positive_lexicon_size": len(positive_words),
        "negative_lexicon_size": len(negative_words),
    }


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------

class NaiveBayes:
    """
    Multinomial-style Naive Bayes over sparse count/binary document vectors.

    For class c:
        P(w|c) = (count(w,c) + 1) / (total_words(c) + |V|)

    Classification:
        argmax_c [ log P(c) + sum_w count(w,d) log P(w|c) ]

    Add-one/Laplace smoothing is used exactly for the word likelihoods.
    """

    def __init__(self, vocabulary_size: int):
        self.V = vocabulary_size
        self.log_prior = [0.0, 0.0]
        self.word_counts = [
            [0.0] * vocabulary_size,
            [0.0] * vocabulary_size,
        ]
        self.total_words = [0.0, 0.0]
        self.log_likelihood = [
            [0.0] * vocabulary_size,
            [0.0] * vocabulary_size,
        ]

    def fit(
        self,
        X: Sequence[Dict[int, float]],
        y: Sequence[int],
    ) -> "NaiveBayes":
        n = len(y)
        if n == 0:
            raise ValueError("Cannot train on an empty dataset.")

        doc_counts = [0, 0]

        for vector, label in zip(X, y):
            doc_counts[label] += 1
            for j, value in vector.items():
                self.word_counts[label][j] += value
                self.total_words[label] += value

        for c in (NEGATIVE, POSITIVE):
            self.log_prior[c] = math.log(doc_counts[c] / n)

            denominator = self.total_words[c] + self.V

            for j in range(self.V):
                probability = (
                    self.word_counts[c][j] + 1.0
                ) / denominator
                self.log_likelihood[c][j] = math.log(probability)

        return self

    def predict_one(self, vector: Dict[int, float]) -> int:
        scores = [self.log_prior[0], self.log_prior[1]]

        for j, value in vector.items():
            scores[0] += value * self.log_likelihood[0][j]
            scores[1] += value * self.log_likelihood[1][j]

        return POSITIVE if scores[POSITIVE] > scores[NEGATIVE] else NEGATIVE

    def predict(
        self,
        X: Sequence[Dict[int, float]],
    ) -> List[int]:
        return [self.predict_one(vector) for vector in X]


# ---------------------------------------------------------------------------
# Logistic Regression + cross-entropy + SGD
# ---------------------------------------------------------------------------

class LogisticRegressionSGD:
    """
    Binary logistic regression.

    z = w.x + b
    p = sigmoid(z)

    Binary cross-entropy:
      -[y log(p) + (1-y) log(1-p)]

    For one training example:
      dL/dw_j = (p-y) x_j
      dL/db   = (p-y)

    SGD update:
      w_j <- w_j - eta * (p-y) * x_j
      b   <- b   - eta * (p-y)

    Zero initialization follows the binary logistic regression presentation
    in Jurafsky & Martin Chapter 4.
    """

    def __init__(
        self,
        vocabulary_size: int,
        learning_rate: float = 0.01,
        epochs: int = 20,
        seed: int = 42,
        shuffle_each_epoch: bool = True,
    ):
        self.V = vocabulary_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.shuffle_each_epoch = shuffle_each_epoch

        self.weights = [0.0] * vocabulary_size
        self.bias = 0.0
        self.loss_history: List[float] = []

    @staticmethod
    def sigmoid(z: float) -> float:
        # Numerically stable sigmoid.
        if z >= 0:
            e = math.exp(-z)
            return 1.0 / (1.0 + e)

        e = math.exp(z)
        return e / (1.0 + e)

    def probability(self, vector: Dict[int, float]) -> float:
        z = self.bias

        for j, value in vector.items():
            z += self.weights[j] * value

        return self.sigmoid(z)

    @staticmethod
    def cross_entropy(probability: float, label: int) -> float:
        # Avoid log(0).
        eps = 1e-15
        p = min(max(probability, eps), 1.0 - eps)

        return -(
            label * math.log(p)
            + (1 - label) * math.log(1.0 - p)
        )

    def compute_loss(
        self,
        X: Sequence[Dict[int, float]],
        y: Sequence[int],
    ) -> float:
        if not y:
            return 0.0

        total = 0.0

        for vector, label in zip(X, y):
            p = self.probability(vector)
            total += self.cross_entropy(p, label)

        return total / len(y)

    def fit(
        self,
        X: Sequence[Dict[int, float]],
        y: Sequence[int],
        verbose: bool = False,
    ) -> "LogisticRegressionSGD":

        if not X:
            raise ValueError("Cannot train on an empty dataset.")

        rng = random.Random(self.seed)
        indices = list(range(len(X)))

        self.loss_history = []

        for epoch in range(1, self.epochs + 1):

            if self.shuffle_each_epoch:
                rng.shuffle(indices)

            for i in indices:
                vector = X[i]
                label = y[i]

                p = self.probability(vector)

                # dL/dz = p-y
                error = p - label

                # Sparse SGD update.
                for j, value in vector.items():
                    gradient = error * value
                    self.weights[j] -= (
                        self.learning_rate * gradient
                    )

                self.bias -= self.learning_rate * error

            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            if verbose:
                print(
                    f"    Epoch {epoch:3d}/{self.epochs} "
                    f"cross-entropy={loss:.6f}"
                )

        return self

    def predict_probability(
        self,
        vector: Dict[int, float],
    ) -> float:
        return self.probability(vector)

    def predict_one(self, vector: Dict[int, float]) -> int:
        return POSITIVE if self.probability(vector) > 0.5 else NEGATIVE

    def predict(
        self,
        X: Sequence[Dict[int, float]],
    ) -> List[int]:
        return [self.predict_one(vector) for vector in X]

    def top_weights(
        self,
        vocabulary: Dict[str, int],
        n: int = 30,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        index_to_word = {i: w for w, i in vocabulary.items()}

        ranked = sorted(
            (
                (index_to_word[i], weight)
                for i, weight in enumerate(self.weights)
                if i in index_to_word
            ),
            key=lambda x: x[1],
        )

        negative = ranked[:n]
        positive = list(reversed(ranked[-n:]))

        return negative, positive


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[int, int, int, int]:
    tn = fp = fn = tp = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == NEGATIVE and predicted == NEGATIVE:
            tn += 1
        elif actual == NEGATIVE and predicted == POSITIVE:
            fp += 1
        elif actual == POSITIVE and predicted == NEGATIVE:
            fn += 1
        else:
            tp += 1

    return tp, tn, fp, fn


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Metrics:
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return Metrics(tp=tp, tn=tn, fp=fp, fn=fn)


def print_metrics(name: str, metrics: Metrics) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"Confusion matrix [[TN, FP], [FN, TP]] = "
        f"[[{metrics.tn}, {metrics.fp}], "
        f"[{metrics.fn}, {metrics.tp}]]"
    )
    print(f"Precision : {metrics.precision:.4f}")
    print(f"Recall    : {metrics.recall:.4f}")
    print(f"Accuracy  : {metrics.accuracy:.4f}")
    print(f"F1        : {metrics.f1:.4f}")


# ---------------------------------------------------------------------------
# Train/dev/test utilities
# ---------------------------------------------------------------------------

def stratified_split(
    reviews: Sequence[Review],
    dev_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Review], List[Review]]:
    """
    Stratified split used only inside an outer training pool.

    This is NOT the final test split.
    """
    rng = random.Random(seed)

    by_class = {
        NEGATIVE: [r for r in reviews if r.label == NEGATIVE],
        POSITIVE: [r for r in reviews if r.label == POSITIVE],
    }

    train: List[Review] = []
    dev: List[Review] = []

    for label in (NEGATIVE, POSITIVE):
        items = list(by_class[label])
        rng.shuffle(items)

        n_dev = max(1, round(len(items) * dev_fraction))

        dev.extend(items[:n_dev])
        train.extend(items[n_dev:])

    rng.shuffle(train)
    rng.shuffle(dev)

    return train, dev


def build_feature_data(
    train_reviews: Sequence[Review],
    test_reviews: Sequence[Review],
    feature: int,
) -> Tuple[Dict[str, int], List[Dict[int, float]], List[Dict[int, float]]]:

    vocabulary = build_vocabulary(
        train_reviews,
        feature=feature,
        include_test=test_reviews if feature == 2 else None,
    )

    X_train = make_vectors(train_reviews, feature, vocabulary)
    X_test = make_vectors(test_reviews, feature, vocabulary)

    return vocabulary, X_train, X_test


# ---------------------------------------------------------------------------
# Optional LR hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_lr_on_dev(
    train_reviews: Sequence[Review],
    dev_reviews: Sequence[Review],
    feature: int,
    learning_rates: Sequence[float],
    epoch_values: Sequence[int],
    seed: int,
    verbose: bool = False,
) -> Tuple[float, int]:
    # Hyperparameters are selected using only the outer training pool.
    # For Feature 2, the inner vocabulary is train + dev; the outer test
    # vocabulary is introduced only when the final outer model is built.
    vocabulary = build_vocabulary(
        train_reviews,
        feature=feature,
        include_test=dev_reviews if feature == 2 else None,
    )

    X_train = make_vectors(train_reviews, feature, vocabulary)
    y_train = [r.label for r in train_reviews]

    X_dev = make_vectors(dev_reviews, feature, vocabulary)
    y_dev = [r.label for r in dev_reviews]

    best = None

    for lr in learning_rates:
        for epochs in epoch_values:
            model = LogisticRegressionSGD(
                len(vocabulary),
                learning_rate=lr,
                epochs=epochs,
                seed=seed,
            )

            model.fit(X_train, y_train, verbose=False)

            dev_pred = model.predict(X_dev)
            metrics = evaluate(y_dev, dev_pred)

            score = metrics.f1

            if verbose:
                print(
                    f"      LR tune: lr={lr:g}, epochs={epochs}, "
                    f"dev F1={score:.4f}"
                )

            candidate = (score, lr, epochs)

            if best is None or candidate > best:
                best = candidate

    assert best is not None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# One outer fold
# ---------------------------------------------------------------------------

def run_fold(
    all_reviews: Sequence[Review],
    test_fold: int,
    feature: int,
    lr: float,
    epochs: int,
    seed: int,
    tune_lr: bool = False,
    verbose: bool = False,
) -> Dict:
    test_reviews = [
        r for r in all_reviews
        if r.fold == test_fold
    ]

    train_pool = [
        r for r in all_reviews
        if r.fold != test_fold
    ]

    if not test_reviews:
        raise ValueError(f"No reviews found for fold {test_fold}.")

    # Optional inner development split for LR.
    chosen_lr = lr
    chosen_epochs = epochs

    if tune_lr:
        inner_train, dev = stratified_split(
            train_pool,
            dev_fraction=0.10,
            seed=seed + test_fold,
        )

        chosen_lr, chosen_epochs = tune_lr_on_dev(
            inner_train,
            dev,
            feature=feature,
            learning_rates=(0.001, 0.005, 0.01, 0.05),
            epoch_values=(10, 20, 40),
            seed=seed + test_fold,
            verbose=verbose,
        )

    # Final models are trained on the COMPLETE outer training pool.
    vocabulary, X_train, X_test = build_feature_data(
        train_pool,
        test_reviews,
        feature,
    )

    y_train = [r.label for r in train_pool]
    y_test = [r.label for r in test_reviews]

    # ---------------- NB ----------------
    nb = NaiveBayes(len(vocabulary))
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_metrics = evaluate(y_test, nb_pred)

    # ---------------- LR ----------------
    lr_model = LogisticRegressionSGD(
        len(vocabulary),
        learning_rate=chosen_lr,
        epochs=chosen_epochs,
        seed=seed + test_fold,
    )

    lr_model.fit(
        X_train,
        y_train,
        verbose=verbose,
    )

    lr_pred = lr_model.predict(X_test)
    lr_metrics = evaluate(y_test, lr_pred)

    return {
        "fold": test_fold,
        "feature": feature,
        "feature_name": FEATURE_NAMES[feature],
        "vocabulary_size": len(vocabulary),
        "train_size": len(train_pool),
        "test_size": len(test_reviews),
        "lr": chosen_lr,
        "epochs": chosen_epochs,
        "test_reviews": test_reviews,
        "y_test": y_test,
        "nb_pred": nb_pred,
        "lr_pred": lr_pred,
        "nb_metrics": nb_metrics,
        "lr_metrics": lr_metrics,
        "lr_model": lr_model,
        "vocabulary": vocabulary,
    }


# ---------------------------------------------------------------------------
# Full 10-fold experiment for one feature
# ---------------------------------------------------------------------------

def run_feature_cv(
    reviews: Sequence[Review],
    feature: int,
    lr: float,
    epochs: int,
    seed: int,
    tune_lr: bool = False,
    verbose: bool = False,
    output_dir: Path | None = None,
) -> Dict:
    """Run one required feature representation over all 10 corpus folds."""
    aggregate_true: List[int] = []
    aggregate_nb: List[int] = []
    aggregate_lr: List[int] = []
    fold_rows: List[Dict] = []
    all_test_reviews: List[Review] = []
    all_nb_predictions: List[int] = []
    all_lr_predictions: List[int] = []

    print("\n" + "=" * 80)
    print(f"FEATURE {feature}: {FEATURE_NAMES[feature]}")
    print("=" * 80)

    for fold in range(1, 11):
        print(f"\nFold {fold}/10")
        result = run_fold(
            all_reviews=reviews,
            test_fold=fold,
            feature=feature,
            lr=lr,
            epochs=epochs,
            seed=seed,
            tune_lr=tune_lr,
            verbose=verbose,
        )

        nb_m = result["nb_metrics"]
        lr_m = result["lr_metrics"]
        print(f"  Vocabulary size: {result['vocabulary_size']:,}")
        print(f"  Train={result['train_size']} Test={result['test_size']}")
        if tune_lr:
            print(
                f"  Chosen LR: learning_rate={result['lr']:g}, "
                f"epochs={result['epochs']}"
            )
        print_metrics("  Naive Bayes", nb_m)
        print_metrics("  Logistic Regression", lr_m)

        aggregate_true.extend(result["y_test"])
        aggregate_nb.extend(result["nb_pred"])
        aggregate_lr.extend(result["lr_pred"])
        all_test_reviews.extend(result["test_reviews"])
        all_nb_predictions.extend(result["nb_pred"])
        all_lr_predictions.extend(result["lr_pred"])

        fold_rows.append({
            "feature": feature,
            "feature_name": FEATURE_NAMES[feature],
            "fold": fold,
            "vocabulary_size": result["vocabulary_size"],
            "train_size": result["train_size"],
            "test_size": result["test_size"],
            "learning_rate": result["lr"],
            "epochs": result["epochs"],
            "nb_precision": nb_m.precision,
            "nb_recall": nb_m.recall,
            "nb_accuracy": nb_m.accuracy,
            "nb_f1": nb_m.f1,
            "nb_tn": nb_m.tn,
            "nb_fp": nb_m.fp,
            "nb_fn": nb_m.fn,
            "nb_tp": nb_m.tp,
            "lr_precision": lr_m.precision,
            "lr_recall": lr_m.recall,
            "lr_accuracy": lr_m.accuracy,
            "lr_f1": lr_m.f1,
            "lr_tn": lr_m.tn,
            "lr_fp": lr_m.fp,
            "lr_fn": lr_m.fn,
            "lr_tp": lr_m.tp,
        })

    nb_all = evaluate(aggregate_true, aggregate_nb)
    lr_all = evaluate(aggregate_true, aggregate_lr)

    # Macro-average of the ten fold metrics, reported in addition to the
    # pooled 2000-document results.
    macro = {
        "nb_precision": mean(r["nb_precision"] for r in fold_rows),
        "nb_recall": mean(r["nb_recall"] for r in fold_rows),
        "nb_accuracy": mean(r["nb_accuracy"] for r in fold_rows),
        "nb_f1": mean(r["nb_f1"] for r in fold_rows),
        "lr_precision": mean(r["lr_precision"] for r in fold_rows),
        "lr_recall": mean(r["lr_recall"] for r in fold_rows),
        "lr_accuracy": mean(r["lr_accuracy"] for r in fold_rows),
        "lr_f1": mean(r["lr_f1"] for r in fold_rows),
    }

    print("\n" + "-" * 80)
    print(f"10-FOLD POOLED RESULTS: FEATURE {feature}")
    print("-" * 80)
    print_metrics("Naive Bayes", nb_all)
    print_metrics("Logistic Regression", lr_all)
    print("\n10-fold macro-average:")
    print(
        f"  NB: Precision={macro['nb_precision']:.4f}, "
        f"Recall={macro['nb_recall']:.4f}, "
        f"Accuracy={macro['nb_accuracy']:.4f}, F1={macro['nb_f1']:.4f}"
    )
    print(
        f"  LR: Precision={macro['lr_precision']:.4f}, "
        f"Recall={macro['lr_recall']:.4f}, "
        f"Accuracy={macro['lr_accuracy']:.4f}, F1={macro['lr_f1']:.4f}"
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_fold_rows(output_dir / f"feature_{feature}_fold_results.csv", fold_rows)
        write_predictions(
            output_dir / f"feature_{feature}_predictions.csv",
            all_test_reviews,
            aggregate_true,
            aggregate_nb,
            aggregate_lr,
        )

    return {
        "feature": feature,
        "feature_name": FEATURE_NAMES[feature],
        "nb": nb_all,
        "lr": lr_all,
        "macro": macro,
        "fold_rows": fold_rows,
        "reviews": all_test_reviews,
        "true": aggregate_true,
        "nb_pred": aggregate_nb,
        "lr_pred": aggregate_lr,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def label_name(label: int) -> str:
    return "positive" if label == POSITIVE else "negative"


def write_fold_rows(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    reviews: Sequence[Review],
    y_true: Sequence[int],
    nb_pred: Sequence[int],
    lr_pred: Sequence[int],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "fold",
            "true_label",
            "nb_prediction",
            "lr_prediction",
            "nb_correct",
            "lr_correct",
            "review_text",
        ])

        for review, actual, nbp, lrp in zip(
            reviews, y_true, nb_pred, lr_pred
        ):
            writer.writerow([
                review.filename,
                review.fold,
                label_name(actual),
                label_name(nbp),
                label_name(lrp),
                int(actual == nbp),
                int(actual == lrp),
                review.text.replace("\n", " "),
            ])


def write_summary(path: Path, results: Sequence[Dict]) -> None:
    """Write the main report-ready results table."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature", "feature_name", "classifier",
            "pooled_precision", "pooled_recall", "pooled_accuracy", "pooled_f1",
            "tn", "fp", "fn", "tp",
            "macro_precision", "macro_recall", "macro_accuracy", "macro_f1",
        ])
        for result in results:
            for classifier in ("nb", "lr"):
                m = result[classifier]
                prefix = "nb_" if classifier == "nb" else "lr_"
                writer.writerow([
                    result["feature"], result["feature_name"],
                    "Naive Bayes" if classifier == "nb" else "Logistic Regression",
                    m.precision, m.recall, m.accuracy, m.f1,
                    m.tn, m.fp, m.fn, m.tp,
                    result["macro"][prefix + "precision"],
                    result["macro"][prefix + "recall"],
                    result["macro"][prefix + "accuracy"],
                    result["macro"][prefix + "f1"],
                ])


# ---------------------------------------------------------------------------
# Q4: Error analysis
# ---------------------------------------------------------------------------

def write_q4_best_model_analysis(
    path: Path,
    best_nb: Dict,
    best_lr: Dict,
    max_examples: int = 10,
) -> None:
    """Compare the separately best NB and LR models on the same 2000 reviews."""
    nb_pred_by_file = {
        r.filename: p for r, p in zip(best_nb["reviews"], best_nb["nb_pred"])
    }
    lr_pred_by_file = {
        r.filename: p for r, p in zip(best_lr["reviews"], best_lr["lr_pred"])
    }
    truth_by_file = {
        r.filename: r.label for r in best_nb["reviews"]
    }
    review_by_file = {r.filename: r for r in best_nb["reviews"]}

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "comparison", "filename", "fold", "true_label",
            "best_nb_prediction", "best_lr_prediction", "review_text"
        ])
        for filename in sorted(review_by_file):
            actual = truth_by_file[filename]
            nbp = nb_pred_by_file[filename]
            lrp = lr_pred_by_file[filename]
            if nbp == actual and lrp != actual:
                comparison = "NB_correct_LR_wrong"
            elif nbp != actual and lrp == actual:
                comparison = "NB_wrong_LR_correct"
            elif nbp != actual and lrp != actual:
                comparison = "both_wrong"
            else:
                comparison = "both_correct"
            r = review_by_file[filename]
            writer.writerow([
                comparison, r.filename, r.fold, label_name(actual),
                label_name(nbp), label_name(lrp), r.text.replace("\n", " ")
            ])

    # Also print a small set of human-readable cases for the report.
    cases = {"NB_correct_LR_wrong": [], "NB_wrong_LR_correct": []}
    for filename in sorted(review_by_file):
        actual = truth_by_file[filename]
        nbp = nb_pred_by_file[filename]
        lrp = lr_pred_by_file[filename]
        if nbp == actual and lrp != actual:
            cases["NB_correct_LR_wrong"].append(review_by_file[filename])
        elif nbp != actual and lrp == actual:
            cases["NB_wrong_LR_correct"].append(review_by_file[filename])

    with path.with_suffix(".txt").open("w", encoding="utf-8") as f:
        f.write("Q4 ERROR ANALYSIS\n")
        f.write("==================\n\n")
        f.write(f"Best NB: Feature {best_nb['feature']} - {best_nb['feature_name']}\n")
        f.write(f"Best LR: Feature {best_lr['feature']} - {best_lr['feature_name']}\n\n")
        for title, items in cases.items():
            f.write(title + "\n" + "-" * len(title) + "\n")
            for r in items[:max_examples]:
                nbp = nb_pred_by_file[r.filename]
                lrp = lr_pred_by_file[r.filename]
                f.write(
                    f"\n[{r.filename}] fold={r.fold} "
                    f"true={label_name(r.label)} "
                    f"NB={label_name(nbp)} LR={label_name(lrp)}\n"
                )
                f.write(r.text.replace("\n", " ")[:2000] + "\n")


def print_error_examples(best_nb: Dict, best_lr: Dict, max_examples: int = 5) -> None:
    nb_pred = {r.filename: p for r, p in zip(best_nb["reviews"], best_nb["nb_pred"])}
    lr_pred = {r.filename: p for r, p in zip(best_lr["reviews"], best_lr["lr_pred"])}
    reviews = {r.filename: r for r in best_nb["reviews"]}

    print("\nQ4 ERROR ANALYSIS EXAMPLES")
    print("=" * 80)
    for title, condition in (
        ("Naive Bayes correct, Logistic Regression wrong", lambda a,n,l: n == a and l != a),
        ("Logistic Regression correct, Naive Bayes wrong", lambda a,n,l: n != a and l == a),
    ):
        print("\n" + title + ":")
        count = 0
        for filename in sorted(reviews):
            r = reviews[filename]
            if condition(r.label, nb_pred[filename], lr_pred[filename]):
                print(
                    f"\n[{filename}] true={label_name(r.label)} "
                    f"NB={label_name(nb_pred[filename])} "
                    f"LR={label_name(lr_pred[filename])}"
                )
                print(r.text.replace("\n", " ")[:1200])
                count += 1
                if count >= max_examples:
                    break


# ---------------------------------------------------------------------------
# Optional LR weight inspection
# ---------------------------------------------------------------------------

def save_lr_weights(
    path: Path,
    model: LogisticRegressionSGD,
    vocabulary: Dict[str, int],
) -> None:
    index_to_word = {i: word for word, i in vocabulary.items()}

    rows = []

    for i, weight in enumerate(model.weights):
        if i in index_to_word:
            rows.append((index_to_word[i], weight))

    rows.sort(key=lambda x: x[1], reverse=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "weight"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Full assignment runner
# ---------------------------------------------------------------------------

def run_all_features(
    reviews: Sequence[Review],
    lr: float,
    epochs: int,
    seed: int,
    tune_lr: bool,
    output_dir: Path,
    verbose: bool,
) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    for feature in range(1, 8):
        results.append(run_feature_cv(
            reviews=reviews,
            feature=feature,
            lr=lr,
            epochs=epochs,
            seed=seed,
            tune_lr=tune_lr,
            verbose=verbose,
            output_dir=output_dir,
        ))

    write_summary(output_dir / "all_features_summary.csv", results)

    # "Best" is defined explicitly as highest pooled F1, separately for NB
    # and LR. This is then used for Q4 as requested by the assignment.
    best_nb = max(results, key=lambda r: (r["nb"].f1, r["nb"].accuracy))
    best_lr = max(results, key=lambda r: (r["lr"].f1, r["lr"].accuracy))

    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS (highest pooled F1)")
    print("=" * 80)
    print(
        f"Best Naive Bayes: Feature {best_nb['feature']} - "
        f"{best_nb['feature_name']} (F1={best_nb['nb'].f1:.4f})"
    )
    print(
        f"Best Logistic Regression: Feature {best_lr['feature']} - "
        f"{best_lr['feature_name']} (F1={best_lr['lr'].f1:.4f})"
    )

    write_q4_best_model_analysis(
        output_dir / "q4_best_model_error_analysis.csv",
        best_nb,
        best_lr,
    )
    print_error_examples(best_nb, best_lr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scratch movie-review sentiment classifier."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to review_polarity/txt_sentoken",
    )

    parser.add_argument(
        "--feature",
        type=int,
        choices=range(1, 8),
        default=None,
        help="Run only one of the seven assignment features.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 11),
        default=None,
        help="Run only one outer fold.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all seven features over all ten folds.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Logistic-regression learning rate for raw count features.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of SGD epochs.",
    )

    parser.add_argument(
        "--tune-lr",
        action="store_true",
        help=(
            "Tune LR learning rate/epochs using an inner development "
            "split of each outer training fold."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Directory for CSV results.",
    )

    parser.add_argument(
        "--bonus-lexicon",
        action="store_true",
        help=(
            "Run the extra-credit experiment: Feature V plus Hu & Liu "
            "positive/negative lexicon count features."
        ),
    )

    parser.add_argument(
        "--lexicon-dir",
        default="hu_and_lin",
        help="Directory containing positive-words.txt and negative-words.txt.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print LR training loss every epoch.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reviews = load_reviews(args.data)

    print(
        f"Loaded {len(reviews)} reviews: "
        f"{sum(r.label == POSITIVE for r in reviews)} positive, "
        f"{sum(r.label == NEGATIVE for r in reviews)} negative."
    )

    folds = Counter(r.fold for r in reviews)

    print(
        "Fold sizes: "
        + ", ".join(
            f"{fold}={folds[fold]}"
            for fold in sorted(folds)
        )
    )

    output_dir = Path(args.output)

    # ---------------------------------------------------------------
    # Extra-credit Hu & Liu lexicon experiment
    # ---------------------------------------------------------------
    if args.bonus_lexicon:
        run_lexicon_bonus_cv(
            reviews=reviews,
            lexicon_dir=args.lexicon_dir,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            verbose=args.verbose,
            output_dir=output_dir,
        )
        print("\nBonus results written to:", output_dir.resolve())
        return

    # ---------------------------------------------------------------
    # One fold + one feature
    # ---------------------------------------------------------------
    if args.feature is not None and args.fold is not None:
        result = run_fold(
            all_reviews=reviews,
            test_fold=args.fold,
            feature=args.feature,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            tune_lr=args.tune_lr,
            verbose=args.verbose,
        )

        print(
            f"\nFEATURE {args.feature}: "
            f"{FEATURE_NAMES[args.feature]}"
        )

        print_metrics(
            "Naive Bayes",
            result["nb_metrics"],
        )
        print_metrics(
            "Logistic Regression",
            result["lr_metrics"],
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        write_predictions(
            output_dir / (
                f"feature_{args.feature}_fold_{args.fold}_predictions.csv"
            ),
            result["test_reviews"],
            result["y_test"],
            result["nb_pred"],
            result["lr_pred"],
        )

        # Save the LR weights for this fold.
        save_lr_weights(
            output_dir / (
                f"feature_{args.feature}_fold_{args.fold}_lr_weights.csv"
            ),
            result["lr_model"],
            result["vocabulary"],
        )

        return

    # ---------------------------------------------------------------
    # One feature + all ten folds
    # ---------------------------------------------------------------
    if args.feature is not None:
        result = run_feature_cv(
            reviews=reviews,
            feature=args.feature,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            tune_lr=args.tune_lr,
            verbose=args.verbose,
            output_dir=output_dir,
        )

        return

    # ---------------------------------------------------------------
    # All seven features
    # ---------------------------------------------------------------
    results = run_all_features(
        reviews=reviews,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        tune_lr=args.tune_lr,
        output_dir=output_dir,
        verbose=args.verbose,
    )

    print("\nResults written to:", output_dir.resolve())


if __name__ == "__main__":
    main()
