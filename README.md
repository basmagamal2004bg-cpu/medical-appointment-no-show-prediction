# Medical Appointment No-Show Prediction

A machine learning project that predicts whether a patient will miss a medical appointment, so clinics can send reminders to the patients most likely to no-show.

## Business Problem
Missed appointments waste doctors' time and clinic revenue. If a clinic knows **before the appointment** which patients are at higher risk of not showing up, it can focus reminders (SMS or calls) on them and manage its schedule better.

**Task:** binary classification (1 = no-show, 0 = attended).

## Dataset
- Source: [Medical Appointment No Shows (Kaggle)](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
- Real-world data: 110,527 appointments, 14 columns (Brazil, 2016).
- Target: `No-show`. About 20% of patients missed their appointment (imbalanced data).
- Features: gender, age, neighbourhood, scholarship, hypertension, diabetes, alcoholism, handicap level, SMS reminder, scheduling and appointment dates.

## Project Steps
1. **Data cleaning:** no missing values or duplicates were found. Removed invalid rows (negative age, age above 100, appointments dated before booking): 13 rows removed, 110,514 remain. Dropped ID columns and the original target column to avoid data leakage.
2. **Exploratory data analysis:** class distribution, summary statistics, correlation heatmap, boxplots, no-show rate by waiting time, age and SMS.
3. **Feature engineering:** new feature `waiting_days` (days between booking and appointment).
4. **Preprocessing:** encoded gender, one-hot encoded neighbourhood (89 features in total), 80/20 stratified train/test split, scaling for Logistic Regression, `class_weight="balanced"` to handle class imbalance.
5. **Modeling:** compared Logistic Regression, Decision Tree and Random Forest.
6. **Tuning:** `RandomizedSearchCV` (3-fold cross-validation, ROC-AUC) on Random Forest.
7. **Deployment:** model saved with Joblib and served through a Streamlit app.

## Key Insights
- **Waiting time is the strongest factor:** no-show rate is 4.6% for same-day appointments and 33% for appointments more than 30 days away.
- Teenagers (13-18) miss appointments most (26%); patients over 60 miss the least (15%).
- SMS reminders look like they increase no-shows overall (27.6% vs 16.7%), but this is misleading: SMS is almost never sent for same-day appointments. Among appointments 8-30 days away, SMS lowers no-shows from 35.6% to 29.1%.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.657 | 0.312 | 0.580 | 0.406 | 0.669 |
| Decision Tree | 0.616 | 0.315 | 0.771 | 0.448 | 0.722 |
| Random Forest | 0.566 | 0.299 | 0.854 | 0.443 | 0.723 |
| **Random Forest (tuned)** | **0.616** | **0.318** | **0.787** | **0.453** | **0.737** |

Best parameters: `n_estimators=200, max_depth=14, min_samples_leaf=5, max_features=0.5`.

The final model is the tuned Random Forest. Accuracy is not the main metric because the data is imbalanced; recall, F1 and ROC-AUC matter more here, since a reminder is cheap compared to an empty appointment slot.

## Limitations
- Data comes from one region and a short period.
- The same patient can appear in several appointments, so the test score may be slightly optimistic.
- Behaviour is hard to predict from these features alone (no transport, weather or patient-history information).
- The risk score is a ranking score, not an exact probability, because the model was trained with class weights.

## Repository Structure
```
├── app.py                      # Streamlit app
├── no_show_prediction.ipynb    # Full analysis and modeling notebook
├── no_show_model.joblib        # Trained model
├── model_columns.joblib        # Feature columns used by the model
├── neighbourhoods.joblib       # Neighbourhood list for the app
└── requirements.txt
```

## Installation and Usage
```bash
pip install -r requirements.txt
streamlit run app.py
```
To reproduce the training, download the dataset from Kaggle, place the CSV next to the notebook, and run `no_show_prediction.ipynb` from top to bottom.

## Live Demo
[Streamlit app](https://medical-appointment-no-show-prediction-n9my3djmdrzmntqmkptvud.streamlit.app/)
