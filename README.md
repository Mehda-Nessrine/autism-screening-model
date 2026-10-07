# 🧩 Autism Screening Model

A machine learning project that estimates the chance that a child shows signs of autism (ASD), based on a short 10-question screening test.

It uses **semi-supervised learning**: only 161 of the 797 records have a label, so the model also learns from the 636 unlabeled ones.

> ⚠️ This is a screening and learning project, **not** a medical diagnosis.

---

## 📊 Data

Two child ASD screening datasets (2017 and 2018) are merged into one.

| Type | Records |
|---|---:|
| Labeled (YES) | 85 |
| Labeled (NO) | 76 |
| Unlabeled | 636 |

**Inputs used:** the 10 screening answers (A1–A10), age, gender, jaundice at birth, and family history of ASD.

---

## 🤖 Models

1. Gaussian Naive Bayes (baseline)
2. Self-Training
3. Label Propagation
4. Label Spreading

---

## ✅ Results

| Model | Accuracy |
|---|:---:|
| Gaussian Naive Bayes | **91%** |
| Self-Training | 85% |
| Label Spreading | 85% |
| Label Propagation | 82% |

Label Propagation and Label Spreading did not miss any ASD case in the test set. The test set is small (33 samples), so these numbers are only a guide.

---

## 🚀 How to run

```bash
# 1. Install the requirements
pip install -r requirements.txt
pip install matplotlib seaborn jupyter

# 2. Open the notebook
jupyter notebook autism_model.ipynb
```

Put `Child-Data2017.csv` and `Child-Data2018.csv` in the same folder as the notebook.

---

## 🔮 Example

```python
result = predict_asd_probability(
    self_training_model, scaler, training_columns,
    A1_Score=1, A2_Score=1, A3_Score=0, A4_Score=0, A5_Score=1,
    A6_Score=1, A7_Score=0, A8_Score=1, A9_Score=0, A10_Score=1,
    age=6, gender='f', jundice='yes', Family_ASD='yes'
)
# {'predicted_class': 'Yes', 'probabilities': {'No (0)': 0.0, 'Yes (1)': 1.0}}
```

---

## 🛠 Built with

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly

