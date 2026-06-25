import streamlit as st
import pickle
import string
import nltk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from lime.lime_text import LimeTextExplainer

ps = PorterStemmer()

# ── Preprocessing ──────────────────────────────────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# ── Load model & vectorizer ────────────────────────────────────────────────────
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# ── Prediction pipeline for LIME ──────────────────────────────────────────────
def predict_proba_pipeline(texts):
    processed = [transform_text(t) for t in texts]
    vectors   = tfidf.transform(processed)
    return model.predict_proba(vectors)


# ── LIME explainer ────────────────────────────────────────────────────────────
explainer = LimeTextExplainer(class_names=["Not Spam", "Spam"])


# ── Feedback helpers ──────────────────────────────────────────────────────────
FEEDBACK_FILE = "feedback.csv"

def save_feedback(message, model_prediction, correct_label, confidence):
    """Append one feedback row to feedback.csv"""
    row = {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message":          message,
        "model_prediction": model_prediction,
        "correct_label":    correct_label,
        "confidence":       round(confidence, 4),
        "was_correct":      model_prediction == correct_label,
    }
    df_new = pd.DataFrame([row])
    if os.path.exists(FEEDBACK_FILE):
        df_new.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(FEEDBACK_FILE, index=False)


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame()


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Email / SMS Spam Classifier")
st.markdown("Classify messages as **Spam** or **Ham** — and understand *why*.")

input_sms = st.text_area("Enter the message", height=150,
                          placeholder="Paste any email or SMS text here…")

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

with st.expander("⚙️ Advanced settings"):
    threshold = st.slider(
        "Spam confidence threshold",
        min_value=0.30, max_value=0.95, value=0.50, step=0.05,
        help="Lower = catch more spam (but more false positives). "
             "Higher = only flag very obvious spam."
    )

# ── Run prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    if not input_sms.strip():
        st.warning("Please enter a message first.")
        st.stop()

    proba     = predict_proba_pipeline([input_sms])[0]
    spam_prob = proba[1]
    is_spam   = spam_prob >= threshold
    prediction_label = "Spam" if is_spam else "Not Spam"

    # Store in session so feedback buttons can access it
    st.session_state["last_message"]    = input_sms
    st.session_state["last_prediction"] = prediction_label
    st.session_state["last_confidence"] = spam_prob
    st.session_state["show_results"]    = True

# ── Show results (persists across feedback button clicks) ──────────────────────
if st.session_state.get("show_results"):
    input_sms       = st.session_state["last_message"]
    prediction_label= st.session_state["last_prediction"]
    spam_prob       = st.session_state["last_confidence"]
    is_spam         = prediction_label == "Spam"

    # Result banner
    if is_spam:
        st.error(f"🚨 **SPAM** — {spam_prob * 100:.1f}% confidence")
    else:
        st.success(f"✅ **Not Spam** — {(1 - spam_prob) * 100:.1f}% confidence it's legitimate")

    # Confidence bar
    st.markdown("#### Confidence")
    bar_col1, bar_col2 = st.columns(2)
    bar_col1.metric("Spam probability", f"{spam_prob * 100:.1f}%")
    bar_col2.metric("Ham probability",  f"{(1 - spam_prob) * 100:.1f}%")

    fig_bar, ax = plt.subplots(figsize=(5, 0.6))
    ax.barh(0, spam_prob,       color="#e63946", height=0.5, label="Spam")
    ax.barh(0, 1 - spam_prob, left=spam_prob, color="#2a9d8f", height=0.5, label="Ham")
    ax.axvline(threshold, color="white", linewidth=1.5, linestyle="--")
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
    fig_bar.patch.set_alpha(0)
    st.pyplot(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Feedback section ───────────────────────────────────────────────────────
    st.markdown("#### 💬 Was this prediction correct?")

    fb_col1, fb_col2 = st.columns(2)

    with fb_col1:
        if st.button("✅ Yes, correct!", use_container_width=True):
            save_feedback(input_sms, prediction_label, prediction_label, spam_prob)
            st.success("Thanks! Feedback saved.")
            st.session_state["feedback_given"] = True

    with fb_col2:
        if st.button("❌ No, it's wrong", use_container_width=True):
            st.session_state["show_correction"] = True

    # Correction selector
    if st.session_state.get("show_correction"):
        correct = st.radio(
            "What should it have been?",
            ["Spam", "Not Spam"],
            horizontal=True,
            index=0 if not is_spam else 1   # suggest opposite
        )
        if st.button("Submit correction"):
            save_feedback(input_sms, prediction_label, correct, spam_prob)
            st.warning(f"Got it — logged as **{correct}**. This helps improve the model!")
            st.session_state["show_correction"] = False
            st.session_state["feedback_given"]  = True

    st.markdown("---")

    # ── LIME explanation ───────────────────────────────────────────────────────
    st.markdown("#### 🔍 Why was this flagged?")
    st.caption(
        "LIME perturbs the input text and observes how the model reacts — "
        "revealing which words pushed the prediction toward Spam (🔴) or Ham (🟢)."
    )

    # Only generate LIME if we don't already have it for this exact message
    if st.session_state.get("lime_message") != input_sms:
        with st.spinner("Generating explanation…"):
            exp = explainer.explain_instance(
                input_sms,
                predict_proba_pipeline,
                num_features=10,
                num_samples=200      # reduced from 500 — faster, still accurate
            )
        st.session_state["lime_message"]      = input_sms
        st.session_state["lime_word_weights"] = exp.as_list()

    word_weights = st.session_state["lime_word_weights"]
    words   = [w for w, _ in word_weights]
    weights = [v for _, v in word_weights]
    colors  = ["#e63946" if v > 0 else "#2a9d8f" for v in weights]

    fig, ax = plt.subplots(figsize=(7, max(3, len(words) * 0.45)))
    ax.barh(words[::-1], weights[::-1], color=colors[::-1])
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_xlabel("Influence on prediction  (+ = spam signal, − = ham signal)")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    spam_patch = mpatches.Patch(color="#e63946", label="Pushes toward Spam")
    ham_patch  = mpatches.Patch(color="#2a9d8f", label="Pushes toward Ham")
    ax.legend(handles=[spam_patch, ham_patch], facecolor="#1e1e2e",
              labelcolor="white", fontsize=8)
    st.pyplot(fig, use_container_width=True)

    # Highlighted text
    st.markdown("#### 🖍️ Message highlighted by impact")
    spam_words = {w for w, v in word_weights if v > 0}
    ham_words  = {w for w, v in word_weights if v < 0}

    highlighted = []
    for token in input_sms.split():
        clean = token.lower().strip(string.punctuation)
        if clean in spam_words:
            highlighted.append(
                f'<span style="background:#e6394655;border-radius:3px;padding:1px 3px;'
                f'border:1px solid #e63946;font-weight:bold">{token}</span>'
            )
        elif clean in ham_words:
            highlighted.append(
                f'<span style="background:#2a9d8f44;border-radius:3px;padding:1px 3px;'
                f'border:1px solid #2a9d8f">{token}</span>'
            )
        else:
            highlighted.append(token)

    st.markdown(" ".join(highlighted), unsafe_allow_html=True)
    st.caption("🔴 Red highlights = spam indicators &nbsp;|&nbsp; 🟢 Green highlights = ham indicators")


# ── Feedback dashboard (sidebar) ───────────────────────────────────────────────
st.sidebar.title("📊 Feedback Dashboard")
df_fb = load_feedback()

if df_fb.empty:
    st.sidebar.info("No feedback collected yet. Make some predictions!")
else:
    total     = len(df_fb)
    correct   = df_fb["was_correct"].sum()
    accuracy  = correct / total * 100

    st.sidebar.metric("Total feedback",  total)
    st.sidebar.metric("Model accuracy",  f"{accuracy:.1f}%")
    st.sidebar.metric("Wrong predictions", total - correct)

    # Accuracy trend over time
    df_fb["timestamp"] = pd.to_datetime(df_fb["timestamp"])
    df_fb["cumulative_accuracy"] = (
        df_fb["was_correct"].expanding().mean() * 100
    )

    fig_trend, ax = plt.subplots(figsize=(3, 1.5))
    ax.plot(df_fb.index, df_fb["cumulative_accuracy"], color="#2a9d8f", linewidth=1.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Feedbacks", fontsize=7, color="white")
    ax.set_ylabel("Accuracy %", fontsize=7, color="white")
    ax.set_facecolor("#0e1117")
    fig_trend.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=6)
    st.sidebar.pyplot(fig_trend, use_container_width=True)

    with st.sidebar.expander("📋 View raw feedback data"):
        st.dataframe(df_fb[["timestamp", "model_prediction",
                             "correct_label", "was_correct", "confidence"]],
                     use_container_width=True)

    # Download button
    st.sidebar.download_button(
        "⬇️ Download feedback.csv",
        data=df_fb.to_csv(index=False),
        file_name="feedback.csv",
        mime="text/csv"
    )