# 📧 Email / SMS Spam Classifier

An end-to-end Machine Learning web application that classifies emails or SMS messages into **Spam** or **Ham (Legitimate)** — with explainable predictions, confidence scoring, user feedback collection, and full experiment tracking across 11 models.

📂 **Repository:** [Nishita6/spam_classification](https://github.com/Nishita6/spam_classification)
🔗 **Live Application:** https://spamclassification6.streamlit.app/

---

## 🚀 Features

- **Real-time Classification** — instantly classify any email or SMS as Spam or Ham
- **Confidence Score + Threshold Control** — see the model's probability score and adjust the spam threshold via a slider (not just a binary yes/no)
- **Explainable AI with LIME** — understand *why* a message was flagged; top 10 words that influenced the prediction shown as a bar chart with the original message highlighted word-by-word
- **User Feedback Loop** — users can flag wrong predictions; every correction is saved to `feedback.csv` with timestamp, confidence, and correct label
- **Feedback Dashboard** — sidebar shows real-time model accuracy based on user corrections, cumulative accuracy trend, and downloadable feedback CSV
- **Robust NLP Preprocessing** — lowercasing, tokenization, stop-word removal, and Porter stemming via NLTK

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| ML & Data | Scikit-learn, Pandas, NumPy, XGBoost |
| NLP | NLTK (tokenization, stopwords, stemming) |
| Explainability | LIME (Local Interpretable Model-agnostic Explanations) |
| Frontend & Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git & GitHub |

---

## 🔍 How Explainability Works

LIME (Local Interpretable Model-agnostic Explanations) works by creating hundreds of perturbed versions of the input message — randomly masking words — and observing how the model's prediction changes. Words whose removal causes a large drop in spam probability are flagged as strong spam signals.

This surfaces insights like:
- `free`, `win`, `prize`, `claim`, `urgent` → strong spam indicators 🔴
- `lunch`, `meeting`, `home`, `tomorrow` → strong ham indicators 🟢

---

## 📁 Repository Structure

```
├── Spam_Classifier_project.ipynb   # EDA, preprocessing, model training, MLflow tracking
├── app.py                          # Streamlit app with LIME explainability + feedback loop
├── model.pkl                       # Trained Naive Bayes model
├── vectorizer.pkl                  # Fitted TF-IDF vectorizer
├── feedback.csv                    # User feedback collected at runtime (auto-generated)
├── requirements.txt                # Dependencies
└── README.md
```

---

## ⚙️ How to Run Locally

**Clone the repo:**
```bash
git clone https://github.com/Nishita6/spam_classification.git
cd spam_classification
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the app:**
```bash
streamlit run app.py
```

**View MLflow experiment dashboard:**
```bash
mlflow ui
# open http://localhost:5000
```

---

## 💡 Key Design Decisions

**Why Naive Bayes over KNN (both have precision=1.0)?**
KNN achieves perfect precision but only 90.8% accuracy — it misses too many actual spam messages. Naive Bayes hits 97.7% accuracy with the same precision, making it the better overall choice.

**Why precision over accuracy as the selection metric?**
A false positive (flagging a legitimate email as spam) is more damaging than a false negative (letting spam through). Precision directly measures this: of all messages flagged as spam, what fraction actually are?

**Why LIME over SHAP?**
LIME works directly on raw text without needing access to model internals, making it model-agnostic and compatible with the TF-IDF + Naive Bayes pipeline without any modifications.
