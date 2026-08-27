# NeuroFive ML Track

Welcome to my Machine Learning Internship repository at **NeuroFive Solutions**.

## Internship Progress

### Week 1 – Titanic Dataset Analysis

#### Task 1: Exploratory Data Analysis (EDA)

* Set up the Python data science environment
* Loaded the Titanic dataset
* Explored the dataset using `.head()`, `.info()`, and `.describe()`
* Identified numerical and categorical features
* Examined missing values
* Performed initial data exploration
* Documented key observations

#### Task 2: Data Cleaning & Visualization

* Handled missing values using appropriate techniques (`fillna()` and `drop()`)
* Justified data cleaning decisions
* Detected outliers using boxplots
* Created visualizations using Matplotlib and Seaborn:

  * Histogram
  * Boxplot
  * Bar Chart
  * Correlation Heatmap
* Analyzed the relationship between different features and passenger survival
* Identified gender as one of the strongest factors influencing survival

## Tools & Technologies

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Git
* GitHub

## Dataset

* Titanic Dataset (Kaggle)

## Learning Outcomes

Through this week's tasks, I learned how to:

* Explore and understand a real-world dataset
* Handle missing data effectively
* Detect outliers using visualizations
* Create meaningful data visualizations
* Interpret patterns and relationships in data before building machine learning models

---

**Machine Learning Internship – NeuroFive Solutions**

### Week 2 – Titanic Survival Classification

#### Task 1: Predict Titanic Survival — First Classification Model

- Built my first supervised machine learning classification model
- Loaded and cleaned the Titanic dataset
- Handled missing values:
  - Filled missing `Age` values using the median
  - Filled missing `Embarked` values using the mode
  - Removed the `Cabin` column due to a large number of missing values
- Selected relevant features for prediction:
  - `Pclass`
  - `Sex`
  - `Age`
  - `SibSp`
  - `Parch`
  - `Fare`
  - `Embarked`
- Used one-hot encoding with `pd.get_dummies()` to convert categorical variables into numerical features
- Split the dataset into training and testing sets using `train_test_split`
- Used an 80/20 training-testing split
- Trained a Logistic Regression classification model using Scikit-learn
- Generated survival predictions on the test dataset
- Evaluated the model using `accuracy_score`
- Created and analyzed a confusion matrix
- Interpreted True Positives, True Negatives, False Positives, and False Negatives

#### Model Performance

- **Algorithm:** Logistic Regression
- **Problem Type:** Binary Classification
- **Train/Test Split:** 80/20
- **Accuracy:** **XX.XX%**

#### Confusion Matrix

The confusion matrix was used to evaluate the model's classification performance by comparing the actual survival status with the model's predicted survival status.

It helped identify:
- Correctly predicted survivors
- Correctly predicted passengers who did not survive
- Survivors incorrectly classified as non-survivors
- Non-survivors incorrectly classified as survivors

## Learning Outcomes

Through Week 2 Task 1, I learned how to:

- Understand the basic workflow of supervised machine learning
- Prepare a dataset for machine learning
- Select relevant features and define a target variable
- Encode categorical variables using one-hot encoding
- Split data into training and testing sets
- Understand the purpose of training and testing data
- Build and train a Logistic Regression classification model
- Generate predictions using a trained model
- Evaluate a classification model using accuracy
- Interpret a confusion matrix
- Understand the difference between correct and incorrect classifications
- Connect data preprocessing and exploratory analysis with machine learning model development



Task 2: House Price Prediction — Regression Model
Introduced the concept of regression, where the goal is to predict continuous numerical values
Used the California Housing dataset available through Scikit-learn
Explored the dataset using:
.head()
.shape
.columns
.info()
.describe()
.isnull().sum()
Selected four relevant features for house price prediction:
MedInc
HouseAge
AveRooms
AveOccup
Defined MedHouseVal as the target variable
Split the dataset into training and testing sets using train_test_split
Used an 80/20 training-testing split
Built and trained a Linear Regression model using Scikit-learn
Generated house price predictions using the trained model
Evaluated the regression model using:
RMSE (Root Mean Squared Error)
R² (R-squared) Score
Compared actual and predicted house prices using a scatter plot
Analyzed the relationship between the selected housing features and house prices
Interpreted the R² score in plain English to understand how well the model explains variations in house prices
Model Performance
Algorithm: Linear Regression
Problem Type: Regression
Dataset: California Housing Dataset
Selected Features: MedInc, HouseAge, AveRooms, AveOccup
Train/Test Split: 80/20
RMSE: X.XXXX
R² Score: X.XXXX

Note: Replace the RMSE and R² values above with the actual values obtained from the Google Colab notebook.

Actual vs Predicted Prices

The model's predictions were visualized using a scatter plot comparing actual house prices with predicted house prices.

Points closer to the diagonal reference line indicate predictions that are closer to the actual house prices. This visualization provided a simple way to evaluate the quality of the regression model.

R² Score Interpretation

The R² score represents the proportion of variation in house prices that can be explained by the selected features in the model. For example, an R² score of 0.57 means that approximately 57% of the variation in house prices is explained by the model. The remaining variation may be influenced by other factors that were not included in the model. Therefore, while the model can capture meaningful patterns in house prices, its performance could potentially be improved by using additional relevant features or more advanced machine learning techniques.

Learning Outcomes

Through Week 2 Task 2, I learned how to:

Understand the difference between classification and regression problems
Work with a real-world housing dataset
Select relevant features for a regression problem
Define features and a continuous target variable
Split data into training and testing sets
Understand the purpose of training and testing data in regression
Build and train a Linear Regression model using Scikit-learn
Generate numerical predictions using a trained model
Evaluate regression performance using RMSE
Understand and interpret the R² score
Visualize actual versus predicted values using a scatter plot
Interpret model performance in plain English
Understand how feature selection affects machine learning model performance
Connect data preprocessing and exploratory analysis with regression model development


## Week 3 – Model Evaluation & Tuning

### Task 1: Model Evaluation & Tuning — Beyond Accuracy

In this task, I revisited my **Titanic Classification Model** from Week 2 and evaluated its performance using multiple classification metrics instead of relying only on accuracy.

### Objectives

* Evaluate the baseline Logistic Regression model using **Accuracy, Precision, Recall, and F1-Score**.
* Use `classification_report` from `sklearn.metrics` for detailed model evaluation.
* Understand why accuracy can be misleading, particularly when dealing with imbalanced datasets.
* Apply **GridSearchCV** for systematic hyperparameter tuning.
* Tune two Logistic Regression hyperparameters:

  * `C`
  * `solver`
* Use **5-fold cross-validation** and F1-score as the optimization metric.
* Compare the original and tuned models using a **Before vs After** performance table.

### Baseline Model Results

The original Logistic Regression model achieved:

* **Accuracy:** 80%
* **Class 0 F1-Score:** 0.85
* **Class 1 F1-Score:** 0.72
* **Class 1 Recall:** 0.67

These results demonstrated why accuracy alone is not sufficient for evaluating a classification model.

### Key Learning

This task helped me understand the importance of **Precision, Recall, and F1-Score** when evaluating classification models. I also learned how `GridSearchCV` can systematically test different hyperparameter combinations instead of relying on manual trial and error.

Hyperparameter tuning was used to investigate whether the Logistic Regression model could achieve better classification performance, providing practical experience with model optimization and evaluation.

### Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook / Google Colab
* GitHub

### Notebook

`Week-03/Week_03_Task_01_Model_Evaluation_Tuning.ipynb`


### Task 2: Customer Churn Prediction — Working with a Business Problem

Built a machine learning solution to predict customer churn using the **Telco Customer Churn** dataset. This task focused on applying machine learning to a real-world business problem and translating model results into actionable business insights.

#### Objectives

- Performed exploratory data analysis (EDA) to understand customer churn patterns.
- Analyzed the relationship between churn and features such as contract type, tenure, monthly charges, internet service, and payment method.
- Identified and handled missing values, including converting `TotalCharges` into a numerical format.
- Examined class imbalance in the `Churn` target variable.
- Encoded categorical variables using **One-Hot Encoding**.
- Used stratified train-test splitting to preserve the churn distribution.
- Trained and evaluated two classification models:
  - **Logistic Regression**
  - **Decision Tree Classifier**
- Evaluated the models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Compared the performance of both models.
- Used Decision Tree `feature_importances_` to identify the **top 3 features driving customer churn**.
- Created a business-focused summary explaining the findings and their potential impact on customer retention.

#### Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Class Imbalance Analysis
5. Categorical Feature Encoding
6. Feature Scaling
7. Train-Test Split
8. Logistic Regression
9. Decision Tree Classification
10. Model Evaluation & Comparison
11. Feature Importance Analysis
12. Business Insights

#### Dataset

**Telco Customer Churn Dataset**

The dataset contains customer demographic information, services, contract details, billing information, tenure, and churn status.

Target variable:

- `Churn` — whether the customer left the company (`Yes`/`No`)

#### Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline/interpretable classification model |
| Decision Tree Classifier | Interpretable tree-based classification model |

#### Evaluation Metrics

Because the dataset contains more non-churned customers than churned customers, accuracy alone is not sufficient for evaluating the models. Therefore, Precision, Recall, and F1-Score were also considered when comparing model performance.

#### Key Business Objective

The primary business objective is to identify customers who are at higher risk of leaving the telecom company. Understanding the factors associated with churn can help businesses develop targeted retention strategies, improve customer satisfaction, and reduce customer loss.

#### Notebook

[Week 3 Task 2 — Customer Churn Prediction](Week-03/Week_03_Task_02_Customer_Churn_Prediction.ipynb)



### Week 4 – Task 1: ML Pipeline & Feature Engineering

#### Task: Build a Proper ML Pipeline with Feature Engineering

In Week 4 Task 1, I built a professional Machine Learning pipeline using the **Titanic dataset** and Scikit-learn.

#### Key Tasks Completed

* Built an end-to-end ML pipeline using `Pipeline` and `ColumnTransformer`
* Applied `StandardScaler` to numerical features
* Applied `OneHotEncoder` to categorical features
* Used `SimpleImputer` to handle missing values
* Created two engineered features:

  * `FamilySize` = `SibSp + Parch + 1`
  * `IsAlone` = Indicates whether a passenger was traveling alone
* Trained a Logistic Regression classification model
* Evaluated the model using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix
* Compared the baseline model with the feature-engineered model
* Saved the final trained pipeline using `joblib`
* Loaded and tested the saved pipeline to verify consistent predictions

#### Machine Learning Pipeline

```text
Raw Data
   ↓
Train/Test Split
   ↓
ColumnTransformer
   ├── Numerical Features → Imputation → StandardScaler
   │
   └── Categorical Features → Imputation → OneHotEncoder
   ↓
Logistic Regression
   ↓
Predictions & Evaluation
```

#### Files

* `Week-04/Week_04_Task_01_ML_Pipeline_Feature_Engineering.ipynb` – Complete Google Colab notebook
* `Week-04/train.csv` – Titanic training dataset
* `Week-04/titanic_final_pipeline.joblib` – Saved final ML pipeline

#### Key Learning Outcomes

This task helped me understand how Scikit-learn Pipelines can combine preprocessing and model training into a single reusable workflow. I also learned how feature engineering can be tested systematically and how pipelines help maintain consistent preprocessing while reducing the risk of data leakage.
