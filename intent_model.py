"""
intent_model.py — Intent classification, the NLP/DL core of CineBot.

Two models, reported side by side:
  * Baseline : TF-IDF (1-2 grams) + Linear SVM  (classical ML reference point)
  * Deep     : Keras  TextVectorization -> Embedding -> Conv1D -> GlobalMaxPool
               -> Dense, trained from scratch (no pretrained download needed,
               so it runs fully offline / reproducibly).

70/15/15 stratified split; macro-F1 is the headline metric (class-balanced).
"""
from __future__ import annotations
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
import json, os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

SEED = 42
random.seed(SEED); np.random.seed(SEED)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def split(df):
    X, y = df["text"].values, df["label"].values
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def run_baseline(tr, va, te, labels):
    (X_tr, y_tr), _, (X_te, y_te) = tr, va, te
    pipe_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr = pipe_vec.fit_transform(X_tr)
    clf = LinearSVC(C=1.0, random_state=SEED)
    clf.fit(Xtr, y_tr)
    pred = clf.predict(pipe_vec.transform(X_te))
    return {
        "accuracy": accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, average="macro"),
    }


def build_keras(vocab_size, n_classes, seq_len=24, emb=64):
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential
    model = Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(vocab_size, emb, mask_zero=False),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run_keras(tr, va, te, labels):
    import tensorflow as tf
    from tensorflow.keras.layers import TextVectorization
    tf.random.set_seed(SEED)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = tr, va, te
    lab2id = {l: i for i, l in enumerate(labels)}
    ytr = np.array([lab2id[l] for l in y_tr])
    yva = np.array([lab2id[l] for l in y_va])
    yte = np.array([lab2id[l] for l in y_te])

    SEQ, VOCAB = 24, 4000
    vec = TextVectorization(max_tokens=VOCAB, output_sequence_length=SEQ,
                            standardize="lower_and_strip_punctuation")
    vec.adapt(X_tr)
    vsize = len(vec.get_vocabulary())

    model = build_keras(vsize, len(labels), seq_len=SEQ)
    Xtr = vec(np.array(X_tr)); Xva = vec(np.array(X_va)); Xte = vec(np.array(X_te))

    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                          restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40,
                     batch_size=32, callbacks=[es], verbose=0)
    pred = model.predict(Xte, verbose=0).argmax(1)
    return {
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro"),
        "epochs_run": len(hist.history["loss"]),
        "report": classification_report(yte, pred, target_names=labels, zero_division=0),
    }


if __name__ == "__main__":
    df = load(_os.path.join(_DATA_DIR, "intents.jsonl"))
    labels = sorted(df["label"].unique())
    tr, va, te = split(df)
    print(f"train={len(tr[0])} val={len(va[0])} test={len(te[0])} | classes={labels}\n")

    base = run_baseline(tr, va, te, labels)
    print(f"BASELINE (TF-IDF + LinearSVC): acc={base['accuracy']:.3f} "
          f"macro-F1={base['macro_f1']:.3f}\n")

    k = run_keras(tr, va, te, labels)
    print(f"DEEP (Keras Conv1D): acc={k['accuracy']:.3f} "
          f"macro-F1={k['macro_f1']:.3f} (epochs={k['epochs_run']})\n")
    print(k["report"])
