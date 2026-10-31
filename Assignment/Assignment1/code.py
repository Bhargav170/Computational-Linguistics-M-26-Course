import re, csv

def parse_srilm_debug1(path):
    """
    Parses -debug 1 output. Each sentence block looks like:
        <sentence text>
        1 sentences, 5 words, 0 OOVs
        0 zeroprobs, logprob= -12.345 ppl= 45.2 ppl1= 60.1
    Returns list of (logprob10, n_words) in file order.
    """
    text = open(path, encoding="utf-8").read()
    pattern = re.compile(
        r'(\d+) sentences,\s*(\d+) words,\s*(\d+) OOVs\s*\n'
        r'\d+ zeroprobs, logprob=\s*(-?\d+\.?\d*)'
    )
    out = []
    for _, n_words, _, logprob in pattern.findall(text):
        out.append((float(logprob), int(n_words)))
    return out

tri  = parse_srilm_debug1("trigram_scores.txt")
bi   = parse_srilm_debug1("bigram_scores.txt")
uni  = parse_srilm_debug1("unigram_scores.txt")

assert len(tri) == len(bi) == len(uni), "sentence count mismatch across models"

LOG10_TO_LN = 2.302585092994046  # multiply if you need natural log instead of base-10

def compute_scores(model, unigram):
    rows = []
    for (logp, n), (logp_u, _) in zip(model, unigram):
        n = max(n, 1)
        total_logp = logp
        avg_logp   = logp / n
        slor       = (logp - logp_u) / n
        rows.append((total_logp, avg_logp, slor))
    return rows

trigram_scores = compute_scores(tri, uni)
bigram_scores  = compute_scores(bi,  uni)

# Write combined CSV: one row per sentence, all 6 scores
with open("q1_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "sent_idx",
        "tri_logP", "tri_avgLogP", "tri_SLOR",
        "bi_logP",  "bi_avgLogP",  "bi_SLOR",
    ])
    for i, ((t1, t2, t3), (b1, b2, b3)) in enumerate(zip(trigram_scores, bigram_scores)):
        w.writerow([i, t1, t2, t3, b1, b2, b3])

print(f"Wrote {len(trigram_scores)} rows to q1_scores.csv")