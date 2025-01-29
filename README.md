# 🏦 Fraud Detection using Machine Learning

## Files Description  
notebooks/fraud_detection.ipynb	Main Jupyter Notebook with fraud detection model implementation.  
data/creditcard.csv	Dataset containing credit card transactions.  
requirements.txt	List of required Python libraries.  
results/	Exported PDF and HTML versions of the Jupyter Notebook.  
**NOTE** The `creditcard.csv` file is publicly available on Kaggle. To download it:  

1. Go to the [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).  
2. Download the `creditcard.csv` file.  
3. Place it in the `data/` folder.  

## 📊 Viewing Results  
To view the results of the fraud detection model, you can open one of the following files located in the `results/` folder:  

- **[fraud_detection PDF.pdf](results/fraud_detection PDF.pdf)** – A PDF version of the analysis.    
- **[fraud_detection.html](results/fraud_detection.html)** – A HTML version of the notebook.  


## 📌 Project Overview
This project aims to detect fraudulent transactions using machine learning models, particularly **LightGBM and XGBoost**, enhanced with **anomaly detection** techniques. The final model helps identify fraudulent transactions while minimizing false positives and false negatives.

## 📂 Dataset
- **Source**: Credit card transactions dataset
- **Features**:
  - `Time`: Timestamp of the transaction
  - `V1-V28`: PCA-transformed features to anonymize sensitive data
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = Normal Transaction, 1 = Fraudulent Transaction)
- **Size**: ~285,000 transactions

## 🔍 Exploratory Data Analysis (EDA)
Key observations:
- Fraudulent transactions account for only **0.17%** of the dataset (highly imbalanced data).
- Distribution analysis shows that **fraudulent transactions are lower in value** compared to normal ones.
- Feature correlation indicates that some PCA-transformed features (`V4`, `V11`, `V17`, etc.) are highly correlated with fraud.

## 🛠️ Data Preprocessing
1. **Handling Imbalance**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance fraud cases.
2. **Feature Scaling**: Standardized `Amount` feature to match the PCA-scaled features.
3. **Anomaly Detection**: Used **Isolation Forest** to generate an `anomaly_score` as a new feature.
4. **Train-Test Split**: Data was split into **80% training / 20% testing**.

## 📊 Model Selection & Evaluation
### **1️⃣ XGBoost Model**
| Metric      | Score  |
|------------|--------|
| Precision  | 0.79   |
| Recall     | 0.86   |
| F1-score   | 0.82   |
| AUC-ROC    | 0.9868 |

### **2️⃣ LightGBM Model** (Final Model)
| Metric      | Score  |
|------------|--------|
| Precision  | 0.62   |
| Recall     | 0.87   |
| F1-score   | 0.72   |
| AUC-ROC    | 0.9717 |

### **📈 Business Implications**

🔹 Reducing False Positives: Helps prevent unnecessary transaction declines for legitimate users.
🔹 Reducing False Negatives: Ensures that fraudulent activities are detected before financial losses occur.
🔹 Scalability: The model can be deployed in real-time fraud detection systems used by financial institutions.

### 🛠️ Technologies Used

Python (Pandas, NumPy, Matplotlib, Seaborn)

Machine Learning (Scikit-learn, XGBoost, LightGBM, Imbalanced-learn)

Data Processing (SMOTE, Isolation Forest, PCA, StandardScaler)

Visualization (Matplotlib, Seaborn, Plotly)

Jupyter Notebook for interactive analysis


🏆 **XGBoost was chosen as the final model due to its superior recall and AUC-ROC scores.**