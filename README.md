# Attrition Model Analysis and Training

This repository contains a machine learning project aimed at analyzing employee attrition and predicting outcomes based on various factors. The project includes data preprocessing, feature engineering, and the creation of a multitask neural network model using TensorFlow/Keras.

---

## Overview

The project focuses on predicting two key targets:
1. **Attrition**: Whether an employee will leave the company.
2. **Department**: The department the employee belongs to.

### Key Steps:
1. **Data Preprocessing**:
   - Cleaning and selecting relevant features.
   - Handling categorical and numerical data.
   - Scaling numerical features using `StandardScaler`.
   - Encoding categorical features using `OneHotEncoder`.

2. **Model Creation**:
   - A shared input layer with two branches for multitask learning.
   - Branch 1 predicts the employee's department.
   - Branch 2 predicts employee attrition.
   - Usage of `softmax` activation for multi-class and binary classification outputs.

3. **Model Training and Evaluation**:
   - The model is trained using the Adam optimizer and categorical cross-entropy loss for both outputs.
   - Performance metrics include accuracy and loss for each branch.

---

## Data

The dataset includes various employee-related attributes such as:
- **Numerical Features**: Age, DistanceFromHome, Education, HourlyRate, etc.
- **Categorical Features**: BusinessTravel, Department, EducationField, etc.

The dataset is loaded from a CSV file hosted online.

---

## Dependencies

To run this project, the following libraries are required:
- Python 3.10+
- TensorFlow
- NumPy
- Pandas
- Scikit-learn

---

## Project Structure

1. **Preprocessing**:
   - Import dependencies.
   - Load the dataset and explore its structure.
   - Select relevant features and preprocess them (encoding, scaling).
   - Split the data into training and testing sets.

2. **Model Creation**:
   - Define a shared input layer.
   - Add multiple dense layers for shared learning.
   - Create two separate branches for predicting `Department` and `Attrition`.

3. **Model Training**:
   - The model is trained for 50 epochs with a batch size of 32.
   - Validation data is used to monitor performance during training.

4. **Evaluation**:
   - Evaluate the model's performance on the test dataset.
   - Print the overall loss, department accuracy, and attrition accuracy.

---

## Results

### Model Performance:
- **Department Output Accuracy**: ~61.9%
- **Attrition Output Accuracy**: ~82.9%

### Observations:
- Attrition prediction performs better than department prediction.
- Accuracy might not be the best metric due to class imbalance in the dataset.

---

## How to Run

1. Clone this repository.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
3. Run 'jupyter notebook attrition.ipynb'
