# Titanic Survival Predictor

A machine learning project that predicts passenger survival on the Titanic using a Decision Tree Classifier. The model was trained on the Kaggle Titanic dataset and achieved 78% precision. All data preprocessing, model training, and evaluation steps are documented in a Colab notebook.

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Model Training](#model-training)  
- [Evaluation](#evaluation)  
- [Usage](#usage)  
- [File Structure](#file-structure)  
- [Contributing](#contributing)  
- [License](#license)  

## Overview

This project demonstrates a complete machine learning workflow:

1. **Data Loading**: Import the Kaggle Titanic dataset.  
2. **Data Cleaning**: Handle missing values and drop irrelevant features.  
3. **Feature Engineering**: Encode categorical variables.  
4. **Model Training**: Train a Decision Tree Classifier.  
5. **Evaluation**: Assess model performance with precision, accuracy, recall, F1 score, classification report, and confusion matrix.  
6. **Documentation**: Analysis and visualizations are provided via a Colab notebook.

## Dataset

- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
- **File**: `titanic.csv`

The dataset contains the following columns:

| Column       | Description                                                           |
|--------------|-----------------------------------------------------------------------|
| PassengerId  | Unique passenger identifier                                           |
| Survived     | Survival (0 = No, 1 = Yes)                                            |
| Pclass       | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)                              |
| Name         | Passenger name                                                        |
| Sex          | Gender                                                                |
| Age          | Age in years                                                          |
| SibSp        | Number of siblings/spouses aboard                                     |
| Parch        | Number of parents/children aboard                                     |
| Ticket       | Ticket number                                                         |
| Fare         | Passenger fare                                                        |
| Cabin        | Cabin number                                                          |
| Embarked     | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)  |

## Preprocessing

- **Dropped Columns**:  
  - `PassengerId`, `Name`, `Ticket`, `Cabin` (irrelevant or excessive missing values)
- **Missing Values**:  
  - `Age`: Imputed with median age  
  - `Embarked`: Imputed with the most frequent port  
- **Encoding**:  
  - `Sex` and `Embarked` encoded using `LabelEncoder`

## Model Training

- **Algorithm**: `DecisionTreeClassifier` from scikit-learn  
- **Data Split**: 80% training, 20% testing (stratified on `Survived`)  
- **Hyperparameters**:  
  ```python
  model = DecisionTreeClassifier(
      criterion='gini',
      random_state=42
  )
  model.fit(X_train, y_train)
