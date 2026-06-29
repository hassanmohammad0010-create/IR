import subprocess
import sys


def install_spacy_model():
    """Download en_core_web_sm (best quality). Falls back gracefully if blocked."""
    print("Downloading spaCy English model (en_core_web_sm) …")
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  ✓ en_core_web_sm installed.")
    else:
        print("  ⚠  Could not download en_core_web_sm.")
        print("     The system will fall back to the lookup lemmatizer via")
        print("     spacy-lookups-data (already installed as a dependency).")
        print("     Quality is slightly lower but the system fully functional.")
        print(f"     Error: {result.stderr.strip()[:200]}")


def install_nltk_data():
    """Download NLTK resources needed by the Stemmer."""
    import nltk

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    print("\nDownloading NLTK resources …")
    for path, name in resources:
        try:
            nltk.data.find(path)
            print(f"  ✓ {name} already present.")
        except LookupError:
            nltk.download(name, quiet=True)
            print(f"  ✓ {name} downloaded.")


def verify_spacy():
    """Verify the spaCy pipeline loads correctly."""
    print("\nVerifying spaCy pipeline …")
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            print(f"  ✓ Loaded 'en_core_web_sm' — pipes: {nlp.pipe_names}")
        except OSError:
            from spacy.lang.en import English

            nlp = English()
            nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
            nlp.initialize()
            print(
                f"  ✓ Loaded blank English + lookup lemmatizer — pipes: {nlp.pipe_names}"
            )

        doc = nlp("running quickly through the beautiful forests")
        tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
        print(f"  ✓ Sample lemmatization: {tokens}")

    except Exception as e:
        print(f"  ✗ spaCy verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_spacy_model()
    install_nltk_data()
    verify_spacy()
    print("\n✓ Setup complete. You can now run scripts/build_dataset.py")
