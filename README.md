# SMS Spam Classifier

A machine learning project that reads an SMS message and predicts whether it is **Spam** or **Not Spam (Ham)**.

The project cleans the text, converts it into numbers using **TF-IDF**, trains and compares several classification models (including an **ensemble**), and saves the best one for later use.

---

## Highlights

- Cleaned the **SMS Spam Collection** dataset (5,572 messages, 403 duplicates removed, 5,169 kept).
- Text preprocessing: lowercasing, punctuation removal, stop-word removal and Porter stemming.
- **Leakage-free** workflow: the data is split first, and TF-IDF is learned only from training data using a scikit-learn `Pipeline`.
- Stratified 80/20 split because the data is imbalanced (only about 12.6% spam).
- Models compared using **precision, recall and F1**, not just accuracy.
- Best model: **Bernoulli Naive Bayes** (tied with the voting ensemble) with **99.03% accuracy and 100% precision**.
- Whole pipeline saved as a single `spam_model.pkl` file.

---

## Project Structure

```
SMS-spam-classifier/
|-- Spam_classifier_.ipynb   # Main notebook (all steps, cell by cell)
|-- spam_utf-8.txt                    # Dataset (label + message)
|-- spam_model.pkl                    # Saved pipeline (TF-IDF + best model)
|-- README.md                         # You are here
```

---

## Dataset

- **Name:** SMS Spam Collection
- **Size:** 5,572 messages (5,169 after removing duplicates)
- **Columns used:** `label` (ham / spam) and `text` (the message)
- **Class balance:** 87.37% ham, 12.63% spam

---

## How It Works

1. **Load and clean:** keep the two useful columns, convert labels to numbers (ham = 0, spam = 1), remove duplicates.
2. **Explore:** class balance, message length and most common words in spam vs ham.
3. **Preprocess text:** lowercase, keep only letters and digits, remove stop words, stem words (`winning` becomes `win`).
4. **Split:** 80% train and 20% test, stratified so both parts have the same spam percentage.
5. **Pipeline:** `TfidfVectorizer (3,000 features)` followed by a classifier.
6. **Compare models:** Multinomial NB, Bernoulli NB, Logistic Regression, Random Forest and a soft Voting ensemble.
7. **Evaluate:** confusion matrix and 5-fold cross-validation on the best model.
8. **Save and predict:** store the pipeline with `pickle` and test it on new messages.

---

## Results

Evaluated on 1,034 held-out test messages:

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| **Bernoulli NB**    | 0.9903   | 1.0000    | 0.9237 | 0.9603 |
| Ensemble (Voting)   | 0.9903   | 1.0000    | 0.9237 | 0.9603 |
| Random Forest       | 0.9845   | 0.9752    | 0.9008 | 0.9365 |
| Multinomial NB      | 0.9807   | 1.0000    | 0.8473 | 0.9174 |
| Logistic Regression | 0.9691   | 0.9806    | 0.7710 | 0.8632 |

**Confusion matrix (Bernoulli NB):**

|                  | Predicted ham | Predicted spam |
|------------------|---------------|----------------|
| **Actual ham**   | 903           | 0              |
| **Actual spam**  | 10            | 121            |

- **Zero false positives:** no genuine message was wrongly marked as spam.
- **10 spam messages were missed** (recall of 92.4%).
- **5-fold cross-validation mean F1: 0.921** (folds ranged from 0.905 to 0.956), so the score is stable across different data splits.

---

## How to Run

**1. Install the requirements**

```bash
pip install pandas numpy matplotlib nltk scikit-learn
```

**2. Open the notebook**

```bash
jupyter notebook spam_classifier_corrected.ipynb
```

Keep `spam_utf-8.txt` in the same folder as the notebook, then run the cells from top to bottom. The first cell downloads the NLTK stop-word list, so an internet connection is needed once.

**3. Use the saved model on your own message**

```python
import pickle, re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def transform_text(text):
    words = re.findall(r'[a-z0-9]+', text.lower())
    words = [w for w in words if w not in stop_words]
    return ' '.join(stemmer.stem(w) for w in words)

with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

message = "Congratulations! You have won a prize. Call now to claim"
cleaned = transform_text(message)
print('SPAM' if model.predict([cleaned])[0] == 1 else 'NOT SPAM')
```

> The model expects text that has already gone through `transform_text`, so always apply it before predicting.

---

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, Matplotlib, NLTK, scikit-learn
- **Tools:** Jupyter Notebook

---

## Limitations

- The dataset contains older (2000s-era) UK spam about prizes and ringtones, so modern phishing messages (for example, "Your account is blocked, click to verify") may not be caught.
- Only about 650 spam examples are available, so the model has limited variety to learn from.
- The model is trained on English text only.

---

## Future Improvements

- Train on newer and larger spam datasets, including phishing messages.
- Tune hyperparameters with `GridSearchCV`.
- Try word embeddings or transformer models (for example, BERT).
- Build a small web app with Streamlit or Flask so anyone can paste a message and get a prediction.

---

## Author

Satyam Kumar Singh