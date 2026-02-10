"""
nltk_intro.py — Introduction to NLTK (Natural Language Toolkit)

This script walks through a classic NLP pipeline step-by-step and explains:
- What each step is
- Why we do it
- A clear example of input -> output

Run:
    python3 -m pip install nltk
    python3 nltk_intro.py
"""

from __future__ import annotations

import string
import random
from typing import List, Tuple, Dict

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


# ---------------------------------------------------------
# 0) Download NLTK resources (only if missing)
# ---------------------------------------------------------
def download_nltk_resources() -> None:
    """
    NLTK relies on external resources (models + dictionaries).
    We download only what is missing to avoid unnecessary downloads.
    """
    from nltk.data import find

    def ensure(resource_name: str, path_hint: str) -> None:
        try:
            find(path_hint)
        except LookupError:
            nltk.download(resource_name, quiet=True)

    # Tokenizers
    ensure("punkt", "tokenizers/punkt")

    # POS taggers
    ensure("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger")

    # Stopwords
    ensure("stopwords", "corpora/stopwords")

    # WordNet (lemmatization)
    ensure("wordnet", "corpora/wordnet")
    ensure("omw-1.4", "corpora/omw-1.4")

    # NER resources
    ensure("maxent_ne_chunker", "chunkers/maxent_ne_chunker")
    ensure("words", "corpora/words")

    # Sentiment (VADER)
    ensure("vader_lexicon", "sentiment/vader_lexicon")


# ---------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------
def title(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def show_io(label_in: str, x_in, label_out: str, x_out) -> None:
    """Small helper: show input -> output in a consistent format."""
    print(f"\n{label_in}:")
    print(x_in)
    print(f"\n{label_out}:")
    print(x_out)


def clean_basic(tokens: List[str], lang: str = "english") -> List[str]:
    """
    Minimal preprocessing:
    - lowercase
    - remove punctuation tokens
    - remove stopwords

    Note:
    - This is good for bag-of-words / TF-IDF.
    - For transformer models, you often do much less manual cleaning.
    """
    sw = set(stopwords.words(lang))
    out: List[str] = []
    for t in tokens:
        t_low = t.lower()
        if t_low in string.punctuation:
            continue
        if t_low in sw:
            continue
        out.append(t_low)
    return out


# =========================================================
# 1) TOKENIZATION
# =========================================================
def demo_tokenization() -> None:
    """
    Tokenization splits raw text into manageable units:
    - sentences
    - words / punctuation

    Why it matters:
    - Most NLP algorithms don't work on raw strings; they work on tokens.
    - Better tokenization improves almost every downstream step.
    """
    title("1) TOKENIZATION: naive split() vs NLTK tokenizers")

    text = "Where is St. Paul located? I don't seem to find it. It isn't in my map."
    print("Raw text:")
    print(text)

    naive = text.split(" ")
    words = word_tokenize(text)
    sents = sent_tokenize(text)

    show_io("Naive split(' ')", naive, "NLTK word_tokenize()", words)
    show_io("NLTK sent_tokenize()", sents, "Number of sentences", len(sents))

    print("\nKey point:")
    print("- split() breaks contractions poorly (don't -> don't) and punctuation sticks to words.")
    print("- NLTK tokenizers handle punctuation and sentence boundaries more reliably.")


# =========================================================
# 2) PREPROCESSING
# =========================================================
def demo_preprocessing() -> None:
    """
    Preprocessing aims to reduce noise and normalize text.

    Typical steps for classic ML:
    - lowercase
    - remove punctuation
    - remove stopwords

    Warning:
    - Stopword removal is NOT always good (e.g. sentiment: 'not good' matters).
    """
    title("2) PREPROCESSING: lowercase + punctuation + stopwords")

    text = "Natural Language Processing is fascinating. I am learning NLP with Python!"
    tokens = word_tokenize(text)
    cleaned = clean_basic(tokens, lang="english")

    show_io("Original tokens", tokens, "Cleaned tokens", cleaned)

    print("\nWhy we do it:")
    print("- It reduces vocabulary size and noise for models like TF-IDF / Naive Bayes / SVM.")
    print("- But for sentiment or transformers, keep more of the original text (often no stopwords removal).")


# =========================================================
# 3) STEMMING
# =========================================================
def demo_stemming() -> None:
    """
    Stemming reduces words to approximate roots (no dictionary).
    Good for:
    - Information retrieval
    - Bag-of-words models

    Downside:
    - Can produce non-words (studies -> studi).
    """
    title("3) STEMMING: Porter vs Lancaster vs Snowball")

    text = "Many smart students are sitting here. Studies studied studying!"
    tokens = word_tokenize(text)

    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer("english")

    porter_out = [porter.stem(w.lower()) for w in tokens]
    lanc_out = [lancaster.stem(w.lower()) for w in tokens]
    snow_out = [snowball.stem(w.lower()) for w in tokens]

    show_io("Tokens", tokens, "Porter stems", porter_out)
    show_io("Lancaster stems (more aggressive)", lanc_out, "Snowball stems", snow_out)

    print("\nTakeaway:")
    print("- Lancaster is often too aggressive.")
    print("- Porter/Snowball are common for classic English NLP pipelines.")


# =========================================================
# 4) POS TAGGING
# =========================================================
def demo_pos_tagging() -> None:
    """
    POS tagging assigns a part-of-speech label to each token:
    NN noun, VB verb, JJ adjective, RB adverb...

    Why:
    - Helps lemmatization (verb vs noun)
    - Enables chunking (noun phrases)
    - Useful for information extraction
    """
    title("4) POS TAGGING: nltk.pos_tag()")

    text = "The quick brown fox jumps over the lazy dog."
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    show_io("Tokens", tokens, "POS-tagged tokens", tagged)

    print("\nTip:")
    print("- In a Python REPL: nltk.help.upenn_tagset() for tag meanings.")


# ---------------------------------------------------------
# Penn Treebank POS -> WordNet POS (for better lemmatization)
# ---------------------------------------------------------
def _is_noun(tag: str) -> bool:
    return tag in ["NN", "NNS", "NNP", "NNPS"]

def _is_verb(tag: str) -> bool:
    return tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

def _is_adverb(tag: str) -> bool:
    return tag in ["RB", "RBR", "RBS"]

def _is_adjective(tag: str) -> bool:
    return tag in ["JJ", "JJR", "JJS"]

def penn2wn(tag: str):
    if _is_adjective(tag):
        return wn.ADJ
    if _is_noun(tag):
        return wn.NOUN
    if _is_adverb(tag):
        return wn.ADV
    if _is_verb(tag):
        return wn.VERB
    return wn.NOUN


# =========================================================
# 5) LEMMATIZATION
# =========================================================
def demo_lemmatization() -> None:
    """
    Lemmatization reduces words to a dictionary form (lemma),
    often using POS info.

    Example:
    - "better" -> "good" (with correct POS/context)
    - "sitting" -> "sit"
    - "are" -> "be"
    """
    title("5) LEMMATIZATION: with vs without POS information")

    lzr = WordNetLemmatizer()
    text = "The striped bats are hanging on their feet for best."
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Without POS: default is noun lemmatization
    no_pos = [lzr.lemmatize(w.lower()) for w in tokens]

    # With POS mapping: better results
    with_pos = [lzr.lemmatize(w.lower(), penn2wn(t)) for (w, t) in tagged]

    show_io("Text", text, "Tokens", tokens)
    show_io("Lemmatized (no POS)", no_pos, "Lemmatized (with POS)", with_pos)

    print("\nWhy POS helps:")
    print("- WordNet needs to know whether a token is a noun/verb/adjective/adverb.")


# =========================================================
# 6) CHUNKING (shallow parsing)
# =========================================================
def demo_chunking() -> None:
    """
    Chunking groups tokens into shallow phrases such as noun phrases (NP).

    We do it with a regex grammar over POS tags.
    """
    title("6) CHUNKING: noun phrases with RegexpParser")

    text = "Dogs or small cats saw Sara, John, Tom, the pretty girl and the big bat."
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    print("POS-tagged tokens:")
    print(tagged)

    grammar = r"""
        NP:       {<DT>?<JJ>*<NN.?>}      # noun phrase
        NounList: {(<NP><,>?)+<CC><NP>}   # NP list separated by conjunction
    """
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(tagged)

    print("\nChunk parse tree (text form):")
    print(tree)

    print("\nHow to read it:")
    print("- (NP ...) marks a noun phrase.")
    print("- (NounList ...) tries to capture 'A, B, C and D' patterns.")


# =========================================================
# 7) NER (Named Entity Recognition)
# =========================================================
def demo_ner() -> None:
    """
    Named Entity Recognition finds entities like:
    - PERSON
    - ORGANIZATION
    - GPE (locations/countries/cities)
    """
    title("7) NER: nltk.ne_chunk()")

    text = "Barack Obama was born in Hawaii and worked at the University of Chicago."
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = nltk.ne_chunk(tagged)

    print("Text:")
    print(text)
    print("\nNE tree:")
    print(tree)

    print("\nNote:")
    print("- ne_chunk returns a tree. Entity labels appear as subtrees (e.g., (PERSON ...)).")


# =========================================================
# 8) N-GRAMS + simple generation
# =========================================================
def demo_ngrams_and_generation() -> None:
    """
    An n-gram is a sequence of n consecutive tokens.
    We build a tiny trigram language model:
    (w1, w2) -> possible next words
    """
    title("8) N-GRAMS: trigram model + tiny text generation")

    text = "It is a simple text. This is a simple text. Is it simple? It is simple!"
    tokens = word_tokenize(text)

    trigrams = list(nltk.ngrams(tokens, 3))
    print("Some trigrams:")
    print(trigrams[:12])

    # Build a context -> next-word list mapping for faster generation
    model: Dict[Tuple[str, str], List[str]] = {}
    for a, b, c in trigrams:
        key = (a.lower(), b.lower())
        model.setdefault(key, []).append(c)

    random.seed(7)  # deterministic output for demos
    out = ["It", "is"]
    max_len = 40

    for _ in range(max_len):
        key = (out[-2].lower(), out[-1].lower())
        options = model.get(key, [])
        if not options:
            break
        out.append(random.choice(options))

    print("\nGenerated text:")
    print(" ".join(out))

    print("\nWhy it matters:")
    print("- Bag-of-ngrams is a strong baseline for text classification.")
    print("- N-gram LMs are the historical foundation of language modeling.")


# =========================================================
# 9) CFG Parsing (constituency parsing)
# =========================================================
def demo_cfg_recursive_descent() -> None:
    """
    CFG parsing builds a constituency tree (NP, VP, PP...).
    Recursive descent is simple but can be slow; it is good for teaching.
    """
    title("9) CFG PARSING: RecursiveDescentParser (educational)")

    grammar = nltk.CFG.fromstring("""
        S -> NP VP
        VP -> V NP | V NP PP
        PP -> P NP
        V -> "saw" | "ate" | "walked"
        NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
        Det -> "a" | "an" | "the" | "my"
        N -> "man" | "dog" | "cat" | "telescope" | "park"
        P -> "in" | "on" | "by" | "with"
    """)

    sent = "Mary saw a dog with my telescope".split()
    parser = nltk.RecursiveDescentParser(grammar)
    trees = list(parser.parse(sent))

    print("Sentence:", sent)
    print("Number of parses:", len(trees))
    print("\nFirst parse tree:")
    print(trees[0])

    print("\nImportant idea:")
    print("- CFGs can be ambiguous. The same sentence may have multiple valid parses.")


def demo_cfg_chart_parser_ambiguity() -> None:
    """
    ChartParser uses dynamic programming and handles ambiguity efficiently.
    Classic ambiguity example:
      "I shot an elephant in my pajamas"
    """
    title("10) CFG PARSING: ChartParser + ambiguity (multiple trees)")

    grammar = nltk.CFG.fromstring("""
        S  -> NP VP
        PP -> P NP
        NP -> Det N | Det N PP | 'I'
        VP -> V NP | VP PP
        Det -> 'an' | 'my'
        N -> 'elephant' | 'pajamas'
        V -> 'shot'
        P -> 'in'
    """)

    sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
    parser = nltk.ChartParser(grammar)
    trees = list(parser.parse(sent))

    print("Sentence:", sent)
    print("Number of parses (ambiguity):", len(trees))

    for i, t in enumerate(trees, start=1):
        print(f"\n--- Parse #{i} ---")
        print(t)

    print("\nExplanation:")
    print("- 'in my pajamas' can attach to the VP (I shot while wearing pajamas)")
    print("  or to the NP (the elephant was in my pajamas).")


def demo_pcfg_viterbi() -> None:
    """
    PCFG adds probabilities to rules.
    ViterbiParser selects the most probable parse when ambiguity exists.
    """
    title("11) PCFG PARSING: ViterbiParser (most probable parse)")

    grammar = nltk.PCFG.fromstring("""
        S -> NP VP [1.0]
        VP -> TV NP [0.6]
        VP -> IV [0.1]
        VP -> VP PP [0.3]
        PP -> P NP [1.0]
        TV -> 'saw' [1.0]
        IV -> 'ate' [1.0]
        NP -> 'Jack' [0.4] | 'telescopes' [0.3] | Det N [0.3]
        Det -> 'the' [1.0]
        N -> 'man' [1.0]
        P -> 'with' [1.0]
    """)

    sent = ['Jack', 'saw', 'telescopes']
    parser = nltk.ViterbiParser(grammar)

    print("Sentence:", sent)
    for tree in parser.parse(sent):
        print(tree)


# =========================================================
# 10) SENTIMENT (VADER)
# =========================================================
def demo_sentiment_vader() -> None:
    """
    VADER is a lexicon + rule-based sentiment analyzer for English.
    It outputs:
    - pos / neu / neg proportions
    - compound score in [-1, +1]
    """
    title("12) SENTIMENT: VADER (rule-based, great baseline)")

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    examples = [
        "The movie was great!",
        "The movie was not great.",
        "I liked the book, especially the ending.",
        "The staff were nice, but the food was terrible.",
        "This is AMAZING!!!",
        "This is okay, I guess."
    ]

    for s in examples:
        scores = sia.polarity_scores(s)
        print(f"\nSentence: {s}")
        print("Scores:", scores)
        print("Interpretation: compound =", scores["compound"])

    print("\nHow to interpret compound:")
    print("- close to +1  -> strongly positive")
    print("- close to -1  -> strongly negative")
    print("- around 0     -> neutral / mixed")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    download_nltk_resources()

    demo_tokenization()
    demo_preprocessing()
    demo_stemming()
    demo_pos_tagging()
    demo_lemmatization()
    demo_chunking()
    demo_ner()
    demo_ngrams_and_generation()
    demo_cfg_recursive_descent()
    demo_cfg_chart_parser_ambiguity()
    demo_pcfg_viterbi()
    demo_sentiment_vader()

    title("Done ✅")
    print("NLTK demo finished.")


if __name__ == "__main__":
    main()