# **Text Classification using SVM on BBC News Dataset**

This project demonstrates a machine learning pipeline for text classification using the **BBC News Dataset**. The dataset contains news articles categorized into five classes: 
**Tech**, **Business**, **Sport**, **Politics**, and **Entertainment**. 

The pipeline includes data preprocessing, feature extraction, dimensionality reduction, model training, and evaluation.

---

## **Table of Contents**

1. Project Overview
2. Features
3. Dataset
4. Methodology
    - Data Preprocessing]
    - Feature Extraction]
    - Feature Selection (PCA)]
    - Model Training and Evaluation]
5. Results
6. Requirements
7. Usage Instructions
8. File Descriptions

---

## **Project Overview**

This project uses an **SVM (Support Vector Machine)** classifier to categorize BBC news articles into five predefined categories. The pipeline implements **TF-IDF** for word frequency representation, along with additional features such as text length and average word length. The dimensionality of features is reduced using **Principal Component Analysis (PCA)**, and the final model is evaluated on a test set.

---

## **Features**

- **TF-IDF Features**: Represent word importance across documents.
- **Text Length**: Total word count in each article.
- **Average Word Length**: Measure of linguistic complexity.
- **Dimensionality Reduction**: PCA to reduce feature dimensions while retaining 95% variance.

---

## **Dataset**

The dataset used in this project is the **BBC News Dataset**, containing 2,225 articles distributed among the following five categories:
- Tech
- Business
- Sport
- Politics
- Entertainment

---

## **Methodology**

### **Data Preprocessing**
- Convert text to lowercase.
- Remove special characters and digits.
- Tokenize sentences into words.
- Remove stopwords (e.g., "the", "is", "and").
- Lemmatize words to their base forms.

### **Feature Extraction**
1. **TF-IDF**: Extracts word frequency features normalized by their importance in the corpus.
2. **Text Length**: The number of words in each article.
3. **Average Word Length**: Average number of characters per word.

### **Feature Selection (PCA)**
- Apply PCA to reduce dimensionality while retaining 95% of variance in the feature set.

### **Model Training and Evaluation**
- Train an **SVM (Support Vector Machine)** model with a linear kernel on the training set (80%).
- Evaluate the model on a development set (10%) and a test set (10%).
- Performance metrics include:
  - Accuracy
  - Precision, Recall, and F1-score

---

## **Results**

### **Model Performance**
- **Overall Accuracy on Test Set**: *e.g., 89%*
- **Detailed Performance**:
  - Precision, Recall, and F1-scores are reported for each category in the classification report.

---

## **Requirements**

- Python 3.x
- Required libraries (install via pip):
  ```bash
  pip install -r requirements.txt
  ```

### **Key Libraries**
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nltk`

---

## **Usage Instructions**

### **Running the Project**

#### 1. Download and Extract Project Files
Make sure the following files are in the same directory:
- `Part2.ipynb`: Jupyter Notebook file containing the full project implementation.
- `bbc-text.csv`: Dataset file containing news articles and their corresponding categories.
- Other supporting files, such as the saved model (`best_svm_model.pkl`) and test results (`test_predictions.csv`).

#### 2. Install Required Libraries
Use the following command to install the necessary Python libraries:
```bash
pip install -r requirements.txt
```

#### 3. Load the Dataset
In the `Part2.ipynb` file, load the dataset `bbc-text.csv`. Ensure the file path is correct:
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('bbc-text.csv')

# Display the first few rows of the data
print(data.head())
```

#### 4. Run the Jupyter Notebook
Open and execute the `Part2.ipynb` file to complete the machine learning pipeline:
```bash
jupyter notebook Part2.ipynb
```

#### 5. View Results
After running the notebook:
- Classification performance will be displayed as a classification report.
- The trained SVM model will be saved as `best_svm_model.pkl`.
- The test set predictions will be saved as `test_predictions.csv`.

---

## **File Descriptions**

- **bbc-text.csv**:
  - Contains raw data sets for text classification, including news text (text column) and corresponding categories (category column).
  - In the project, this file will be loaded as a Pandas data box for feature extraction, model training, and testing.
- **Part2.ipynb**:
  - Jupyter Notebook containing the full implementation of the machine learning pipeline.

---

