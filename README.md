# 🧬 Breast Cancer Classification using Neural Network

This project uses a **Neural Network (Deep Learning)** model to classify breast cancer tumors as **Benign or Malignant** based on diagnostic features.

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should **NOT** be used for real medical diagnosis.

---

## 📌 Objective

The objective of this project is to:

* Build a **binary classification model** using neural networks
* Predict whether a tumor is **benign or malignant**
* Evaluate the model using classification metrics
* Provide a simple interface using **Streamlit**

---

## 📊 Dataset

The dataset contains **569 samples** with **30 numerical features** computed from digitized images of breast masses.

### Features include:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension
  *(each with mean, standard error, and worst values)*

### Target Variable:

* **0 → Malignant**
* **1 → Benign**

---

## ⚙️ Project Workflow

1. Data loading
2. Data preprocessing
3. Label encoding
4. Feature scaling using StandardScaler
5. Train-test split (with stratification)
6. Neural network model creation
7. Model training
8. Model evaluation
9. Confusion matrix & classification report
10. Model saving
11. Streamlit app development

---

## 🧠 Model Architecture

* Input Layer: 30 features
* Hidden Layer: Dense (ReLU activation)
* Output Layer: Dense (Softmax activation)

---

## 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🏆 Results

* Model achieved approximately:

```text
Accuracy ≈ 96% – 99%
```

* Good performance on both classes (benign and malignant)

---

## 💾 Model Saving

The trained model and scaler are saved for deployment:

```python
model.save("breast_cancer_nn_model.keras")

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
```

---

## 🖥️ Streamlit App

A Streamlit web app is included.

### Features:

* Upload CSV file with 30 features
* Model predicts tumor type
* Displays prediction with confidence
* Download results as CSV

### Run the app:

```bash
streamlit run app.py
```



