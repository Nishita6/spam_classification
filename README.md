# Email Spam Classifier 📧🚫

An end-to-end Machine Learning web application that classifies emails or SMS messages into **Spam** or **Ham (Legitimate)** with high accuracy. This project covers the entire data science pipeline—from raw text preprocessing and Exploratory Data Analysis (EDA) to model training, evaluation, and deployment.

📂 **Repository:** [Nishita6/spam_classification](https://github.com/Nishita6/spam_classification)

---

## 🚀 Features
* **Real-time Classification:** Input any text block or email body to instantly check if it's spam.
* **Interactive UI:** A clean, user-friendly frontend interface built for seamless interaction.
* **Robust Preprocessing:** Handles text cleaning, tokenization, stop-word removal, and stemming to optimize feature extraction.
* **Data Insights:** Comprehensive Exploratory Data Analysis (EDA) visualizing word frequencies, character counts, and class distributions.

---

## 🛠️ Tech Stack & Tools
* **Language:** Python 🐍
* **Data Analysis & ML:** Jupyter Notebook, Pandas, NumPy, Scikit-Learn
* **Natural Language Processing (NLP):** NLTK (Natural Language Toolkit)
* **Model Deployment:** Streamlit / Flask (Backend), Vercel (Hosting)
* **Version Control:** Git & GitHub

---

## 📊 Methodology & Workflow

### 1. Data Cleaning & Preprocessing
* Removed duplicate entries and handled missing values.
* Converted text to lowercase and tokenized sentences into individual words.
* Stripped special characters, punctuation, and English stop words.
* Applied **Stemming** (PorterStemmer) to reduce words to their base forms.

### 2. Exploratory Data Analysis (EDA)
* Analyzed the distribution of Spam vs. Ham classes to inspect dataset balance.
* Computed statistics on the number of characters, words, and sentences per message.
* Built word clouds and frequency histograms to identify the most common terms in both categories.

### 3. Vectorization & Feature Engineering
* Converted processed text data into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to capture contextual importance.

### 4. Model Training & Selection
* Trained and evaluated multiple classification algorithms (e.g., Naive Bayes, Logistic Regression, Support Vector Machines).
* Tuned hyperparameters to optimize precision and recall, prioritizing low false-positive rates (ensuring important emails aren't accidentally marked as spam).

---

## 📁 Repository Structure
```text
├── .idea/                          # IDE configuration files
├── Spam_Classifier_project.ipynb   # Jupyter Notebook containing EDA, preprocessing, and model training
├── app.py                          # Application source code for deployment
├── model.pkl                       # Trained machine learning model artifact
├── vectorizer.pkl                  # Fitted TF-IDF vectorizer artifact
└── README.md                       # Project documentation
```

---

## ⚙️ How to Run Locally
* Clone the repository:
 ```text
git clone https://github.com/Nishita6/spam_classification.git
cd spam_classification
```
* Install dependencies
``` text
pip install pandas scikit-learn nltk streamlit
```
* Run the application:
``` text
streamlit run app.py
```
<<<<<<< HEAD
=======

>>>>>>> 437576561b169958cd8a4c39acdb6214a5c6c763
