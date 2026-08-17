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

