# CineBot 🎬 — A Domain-Oriented Chatbot for Movie Recommendations

Capstone project (AI / ML / DL, Batch 7). CineBot understands a free-text user message,
figures out what the user wants (intent + mood/genre/era), and recommends movies — all in a
single, self-contained, reproducible Jupyter notebook.

```
user message ─► Intent classifier (DL: Keras Conv1D) ─┐
             ─► Entity extractor (mood/genre/era)     ─┤─► Dialogue manager ─► reply
                                                       └─► Content-based recommender (TF-IDF)
```

## What's in here

| File | Purpose |
|---|---|
| `CineBot_Capstone.ipynb` | **The deliverable.** End-to-end notebook with saved outputs. |
| `requirements.txt` | Python dependencies. |
| `src/` | The same logic as standalone, unit-tested modules (optional; the notebook is self-contained). |
| `data/` | Auto-populated on first run (the notebook downloads the TMDB catalogue and generates the intent corpus). |

## The system, briefly

1. **Data** — TMDB catalogue (10,866 movies → 6,141 after quality cleaning), downloaded from a
   public GitHub mirror. The intent corpus (1,960 balanced utterances over 7 intents) is
   *generated* in-notebook by expanding templates with mood/genre/era slot-fillers.
2. **Intent classification** — two models compared: a **TF-IDF + Linear SVM** baseline and a
   **Keras Embedding → Conv1D** deep model trained *from scratch* (no pre-trained download, so it
   runs fully offline).
3. **Entity extraction** — transparent gazetteer + regex for mood, genre, era and minimum rating
   (with unit tests).
4. **Recommender** — content-based TF-IDF + cosine similarity with slot-aware hard-filtering and a
   Bayesian (IMDB-style) quality prior. Supports "more like *X*".
5. **Dialogue manager** — ties it together with per-session slot memory; `refine` reuses slots and
   skips already-shown titles.

## Results (honestly reported)

| Component | Metric | Result |
|---|---|---|
| Intent — baseline (TF-IDF+SVM) | test macro-F1 | ~1.00 (in-distribution) |
| Intent — deep (Keras Conv1D) | test macro-F1 | ~0.99 (in-distribution) |
| Intent — deep, **unseen** phrasings | accuracy | **~0.69** (realistic) |
| Recommender | genre precision@5 / NDCG@5 | ~1.00 (by content filter) |
| Recommender | mood-appropriateness@5 | ~1.00 |
| Recommender | similar-genre coherence@5 | ~0.61 |

The high in-distribution intent scores partly reflect that the corpus is template-generated; the
**~0.69 on unseen phrasings is the realistic generalisation number**, and the recommender's
genre precision is ~1.0 *by construction* of the hard filter. These caveats are stated plainly in
the notebook — see §10 and §14.

---

## How to run — Option A: Google Colab (easiest, recommended)

1. Go to <https://colab.research.google.com> → **File ▸ Upload notebook** → choose
   `CineBot_Capstone.ipynb`.
2. **Runtime ▸ Run all** (`Ctrl/Cmd + F9`).
3. That's it. Colab already has TensorFlow, scikit-learn, pandas, etc. The notebook downloads its
   own data and trains in ~1–2 minutes on a free CPU runtime. A GPU is **not** required.

## How to run — Option B: Local machine

```bash
# 1. (recommended) create a clean environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3a. launch and run interactively
jupyter notebook CineBot_Capstone.ipynb
#    then: Kernel ▸ Restart & Run All

# 3b. …or execute headless end-to-end (saves outputs back into the file)
jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 CineBot_Capstone.ipynb
```

Requirements: Python 3.10–3.12 and an internet connection on first run (to download the dataset).
Runs on CPU; total runtime is a couple of minutes.

## How to run — Option C: the modular source (optional)

Each component can be exercised on its own:

```bash
cd src
python data_prep.py        # clean catalogue + mood/era derivation
python make_intents.py      # regenerate the intent corpus
python entities.py          # entity extraction unit tests (5/5)
python recommender.py       # recommender demo + evaluation
python intent_model.py      # train baseline + deep model, print metrics
python dialogue.py          # end-to-end conversation (keyword intent stub)
```

## Reproducibility notes

- All seeds are fixed (`SEED = 42`) for NumPy, Python `random`, and TensorFlow.
- The notebook trains from scratch — no external model weights are downloaded, only the CSV
  dataset (cached locally after the first run as `data/tmdb_movies.csv`).
- Re-running **Restart & Run All** reproduces every number and figure.

## Known limitation / future work

The deep model is trained on template-generated text, so it generalises only moderately to
free-form phrasing (~0.69). The clearest upgrade is to **fine-tune a pre-trained transformer
(e.g. DistilBERT)** for intent and to **collect real user utterances** — both are straightforward
on Colab and described in the notebook's conclusions.
