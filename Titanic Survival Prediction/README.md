# Titanic Survival Prediction

This project shows my first step into the **Machine Learning**.This project is about predict survival of passengers on the Titanic dataset.  

## Key Steps
1. **Data Preprocessing**
   - Impute missing values (`Age` with median, `Fare` with most frequent)
   - One-hot encode categorical features (`Sex`, `Embarked`)
   - MinMax scaling  

2. **Model**
   - Decision Tree Classifier ()  

3. **Pipeline**
   - Preprocessing + Model wrapped inside a `Pipeline` for clean workflow  

## Usage
pip install -r requirements.txt
cd src
python pipeline.py
