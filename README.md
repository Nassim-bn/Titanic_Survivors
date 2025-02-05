# Titanic Survival Prediction

## Project Description
The challenge is to predict which passengers survived the Titanic disaster based on their characteristics, such as name, age, ticket price, and class. By analyzing the provided data, the goal is to build a model that determines whether a passenger survived or not.

## The Data
This project uses three main files:

1. **train.csv**
   - Contains detailed information about 891 passengers of the Titanic.
   - Each row represents a passenger, with columns describing their features (e.g., age, sex, class) and a binary `Survived` column (1 = survived, 0 = did not survive).
   - Used to train the model and identify survival patterns.

2. **test.csv**
   - Contains similar information to `train.csv` for 418 passengers, but without the `Survived` column.
   - The goal is to predict the survival of these passengers using the patterns found in `train.csv`.

3. **gender_submission.csv**
   - An example submission file containing:
     - `PassengerId` (IDs from `test.csv`)
     - `Survived` (predictions: 1 = survived, 0 = did not survive)
   - Assumes that all female passengers survived and all male passengers did not.

## Objective
- Build a machine learning model to predict passenger survival.
- Submit predictions in a file structured like `gender_submission.csv`.
- Evaluate model performance based on prediction accuracy.

## Motivation
This project serves as an excellent introduction to machine learning, providing hands-on experience with data exploration, pattern recognition, and predictive modeling.

## Environment Setup
To set up the environment and load essential libraries for data analysis:

```python
import numpy as np
import pandas as pd
import os

# Verify data files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

## Loading the Titanic Dataset

```python
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
```

## Loading the Titanic Test Dataset

```python
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
```

## Gender-Based Survival Analysis

```python
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)
print("% of women who survived:", rate_women)
```

```python
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)
print("% of men who survived:", rate_men)
```

From the analysis, around **75% of women** survived compared to only **19% of men**, indicating that gender is a strong predictor of survival.

## Building a Machine Learning Model: Random Forest

We use a **Random Forest Classifier**, which consists of multiple decision trees, to predict survival.

### Model Training and Prediction

```python
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```

## Conclusion
At the end of this process, the model has made predictions for the test data, which are stored in a CSV file called `submission.csv`. This file contains the predicted survival outcomes for each passenger in the test set.

### Submission Output Example
| PassengerId | Survived |
| ----------- | -------- |
| 892         | 0        |
| 893         | 1        |
| 894         | 0        |
| 895         | 0        |
| 896         | 1        |

This project provides a foundational understanding of data analysis and machine learning techniques, setting the stage for more advanced predictive modeling tasks.

