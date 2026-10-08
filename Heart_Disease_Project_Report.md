# Heart Disease Prediction Project Report

## 1. Project Overview

This project predicts whether a patient may have heart disease using clinical health information and a trained machine-learning model.

The application allows users to enter:

- Age
- Gender
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- ST slope

After clicking **Analyze Heart Risk**, the application returns:

- Heart health score
- Model confidence
- Higher-risk or lower-risk result
- Possible risk factors
- Personalized recommendations
- Medical disclaimer

This application is intended for educational and demonstration purposes. It is not a medical diagnosis system.

## 2. Technologies Used

### Frontend

The frontend is created inside `app.py` using:

- Python Streamlit
- HTML markup
- CSS styling

Streamlit generates the web interface from Python code. Custom HTML and CSS provide the page layout, form styling, buttons, colors, cards, responsive mobile design, input focus effects, and risk-result panels.

There is no separate React, Angular, Vue, or JavaScript frontend.

### Backend

The backend is implemented in Python using:

- Streamlit application server
- Pandas for input data processing
- Joblib for loading saved machine-learning artifacts
- Scikit-learn model artifacts for prediction
- StandardScaler preprocessing

There is no separate Flask, Django, FastAPI, or REST API layer. Streamlit handles the user interface and backend prediction flow together.

## 3. Project Files

### `app.py`

The main application file contains:

- Streamlit page configuration
- Custom CSS and HTML
- Patient input form
- Input validation
- Feature encoding
- Scaling
- Model prediction
- Confidence calculation
- Personalized risk precautions
- Result display

### `heart.csv`

The training dataset contains 5,000 patient records after reproducible augmentation. It preserves the original 918 patient records and adds 4,082 bounded synthetic records generated from the training portion of the original data.

The original dataset contained 918 records. The augmented dataset is intended for project experimentation and is not equivalent to collecting 5,000 new clinical patients.

Original columns:

```text
Age
Sex
ChestPainType
RestingBP
Cholesterol
FastingBS
RestingECG
MaxHR
ExerciseAngina
Oldpeak
ST_Slope
HeartDisease
```

### `HeartdiseaseFinal.ipynb`

The Jupyter Notebook contains:

- Exploratory data analysis
- Data preprocessing
- Train-test splitting
- Feature scaling
- Multiple model training
- Model comparison
- Model export

### `knn_heart_model.pkl`

Saved K-Nearest Neighbors classification model used by the application.

### `heart_scaler.pkl`

Saved `StandardScaler` used to normalize input features before prediction.

### `heart_columns.pkl`

Saved list of expected model feature columns.

### `augment_and_retrain.py`

Reproducible script that:

- Splits the original data into training and test portions.
- Generates 4,082 bounded synthetic training records.
- Produces exactly 5,000 total rows.
- Trains the KNN model on the augmented dataset.
- Compares the augmented model with a baseline using the untouched original test set.
- Saves the updated model, scaler, and feature-column artifacts.

### `heart_augmented.csv`

Copy of the final 5,000-row augmented dataset. The application uses `heart.csv`; this file is retained as an explicit copy of the augmented data for inspection.

## 4. Dataset Details

The final dataset contains:

```text
Total records: 5,000
Input columns: 11 including target
Target column: HeartDisease
```

Dataset construction:

```text
Original records: 918
Synthetic records added: 4,082
Final records: 5,000
```

Target distribution:

```text
HeartDisease = 0: 2,175 records
HeartDisease = 1: 2,825 records
```

The target value means:

- `0`: No heart disease recorded
- `1`: Heart disease recorded

## 5. Machine-Learning Workflow

### Step 1: Load the dataset

The notebook loads `heart.csv` using Pandas.

### Step 2: Separate features and target

The target column is `HeartDisease`. The remaining clinical columns are used as input features.

### Step 3: Encode categorical values

Categorical variables such as gender, chest pain type, resting ECG, exercise angina, and ST slope are converted into numerical one-hot encoded columns.

The deployed model expects 15 features:

```text
Age
RestingBP
Cholesterol
FastingBS
MaxHR
Oldpeak
Sex_M
ChestPainType_ATA
ChestPainType_NAP
ChestPainType_TA
RestingECG_Normal
RestingECG_ST
ExerciseAngina_Y
ST_Slope_Flat
ST_Slope_Up
```

### Step 4: Split the data

The notebook uses an 80/20 train-test split. The split is stratified so the target distribution remains similar in both sets.

### Step 5: Scale numerical features

The notebook uses `StandardScaler`. The scaler is fitted on the training data and applied to the training data, testing data, and new user input.

### Step 6: Train multiple models

The notebook compares:

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine with RBF kernel

Each model is evaluated using accuracy and F1 score.

### Step 7: Select the deployed model

The deployed application uses a K-Nearest Neighbors classifier with:

```text
n_neighbors = 5
weights = uniform
```

## 6. Dataset Augmentation and Evaluation

Because the original dataset contained 918 records, 4,082 additional synthetic records were generated to reach the requested 5,000 rows. Synthetic rows were created by sampling only from the original training split and applying bounded random variation to numeric fields. Categorical values and target labels were inherited from the sampled training records.

The original test split was kept untouched for a fair comparison:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Original-data KNN baseline | 88.59% | 89.95% |
| Augmented-data KNN | 90.22% | 91.26% |

These results show an improvement on this fixed test split, but they do not prove clinical or real-world accuracy. Synthetic data can introduce assumptions and should not be treated as a replacement for additional real patient records.

## 7. Prediction Process

When the user submits the form, the application:

1. Reads the entered patient values.
2. Converts text values into integers or decimal numbers.
3. Validates the allowed ranges.
4. Creates a row containing all expected model features.
5. Encodes the selected categorical values.
6. Applies the saved scaler.
7. Sends the processed row to the KNN model.
8. Generates the prediction.
9. Calculates model probability.
10. Displays the result and personalized precautions.

Prediction flow:

```text
User input
    -> Validation
    -> Categorical encoding
    -> Feature alignment
    -> StandardScaler transformation
    -> KNN prediction
    -> Risk result and recommendations
```

## 8. Frontend Design

The application includes a professional clinical-style interface with:

- Light blue medical theme
- Centered content area
- Patient information card
- Two-column desktop form
- Responsive mobile layout
- Blue gradient action button
- Custom input borders
- Visible input focus states
- Risk result cards
- Confidence and health-score metrics
- Higher-risk and lower-risk color indicators

The result panel displays:

- Prediction Result
- AI-based Heart Disease Analysis
- Heart Health Score
- Model Confidence
- Risk classification
- Possible risk factors
- Recommended actions
- Medical disclaimer

## 9. Personalized Precautions

The precautions are generated from the values entered by the user.

### High fasting blood sugar

The application can display an elevated fasting blood sugar warning and recommend discussing blood sugar and diet with a healthcare professional.

### Exercise angina

The application can display an exercise-induced angina warning and recommend avoiding strenuous exercise until cleared by a doctor.

### High cholesterol

For cholesterol readings of 240 or higher, the application can display a high cholesterol warning and recommend discussing cholesterol management and heart-healthy nutrition with a doctor.

### Elevated resting blood pressure

For resting blood pressure of 140 or higher, the application can recommend rechecking blood pressure and discussing blood-pressure management with a doctor.

### Lower maximum heart rate

For a maximum heart rate below 100, the application can recommend discussing the result with a healthcare professional before strenuous activity.

### Elevated Oldpeak

For Oldpeak values of 2 or higher, the application can recommend professional review of the exercise-related ECG finding.

### Downward ST slope

For a Downward ST slope, the application can recommend professional review of the ECG pattern.

These precautions are rule-based educational guidance. They are not medical diagnoses and should not replace professional medical advice.

## 10. Input Validation

The application validates the following ranges:

```text
Age: 18 to 100
Resting blood pressure: 1 to 250
Cholesterol: 1 to 700
Maximum heart rate: 1 to 250
Oldpeak: -5 to 10
```

Invalid values generate a user-friendly error message instead of sending incorrect data to the model.

## 11. How to Run the Project

Open PowerShell in the project directory and run:

```powershell
cd "C:\Users\adity\OneDrive\Desktop\Machine-Learning-Part-3"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the application at:

```text
http://localhost:8501
```

If port `8501` is unavailable, run:

```powershell
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## 12. Current Limitations

- The system is an educational project, not a certified medical tool.
- The result depends on the quality and representativeness of the training data.
- The augmented dataset includes synthetic records and is not equivalent to 5,000 independent clinical patient records.
- The reported accuracy and F1 score come from one fixed original test split and require further validation.
- The recommendation rules use manually defined thresholds.
- No user accounts or database are included.
- Prediction history is not stored.
- No separate production API exists.
- Model performance metrics should be added to the documentation for a complete evaluation report.
- The application does not replace examination, diagnosis, or treatment by a healthcare professional.

## 13. Conclusion

This project demonstrates a complete machine-learning application workflow. It combines a clinical dataset, preprocessing, model training, saved model artifacts, a Streamlit frontend, real-time prediction, and personalized educational recommendations.

The system is suitable for academic demonstration and project presentation. For real-world clinical use, it would require extensive medical validation, stronger security, formal testing, explainable model analysis, privacy protections, and approval from relevant healthcare authorities.
