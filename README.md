# 🌐 Live Demo

You can try the app here: [Titanic Survival Predictor](#)  
*(Replace `#` with your deployed Streamlit app link if hosted online)*

# 🚢 Titanic Survival Prediction App

This project is a **Machine Learning-based Titanic Survival Prediction System** built with **Python, Scikit-learn, and Streamlit**.  
It predicts whether a passenger survived the Titanic disaster using historical passenger data.

---

## 🚀 Project Overview

The project has two main components:

1. **Model Training (`train_model.py`)**  
   - Performs Exploratory Data Analysis (EDA)  
   - Cleans and preprocesses the dataset  
   - Performs feature engineering (Title, TicketType, etc.)  
   - Trains multiple ML models:  
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)  
     - Naive Bayes  
     - Decision Tree  
     - Support Vector Machine (SVM)  
   - Evaluates models using Accuracy & F1 Score  
   - Saves the best model (`model.pkl`), scaler (`scaler.pkl`), and column names (`columns.pkl`)  

2. **Web App (`app.py`)**  
   - Built with **Streamlit**  
   - Collects passenger details via a sidebar  
   - Preprocesses input using the saved scaler and feature columns  
   - Predicts Titanic survival (Survived / Not Survived)  
   - Displays results clearly using metrics and Streamlit messages  

---

## 📊 Dataset

- **File**: `train.csv`  
- **Rows**: 891 passengers  
- **Target Variable**: `Survived` (0 = Not Survived, 1 = Survived)  
- **Features**:  
  - Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked  
- **Feature Engineering**:  
  - `Title` extracted from Name  
  - `TicketType` extracted from Ticket  
  - `Sex` mapped to numeric  
  - Missing values imputed for Age, Embarked, Cabin  

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/titanic-survival-predictor.git
   cd titanic-survival-predictor

2. Create a virtual environment (optional but recommended):
   ```bash 
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

3. Install required libraries:
   ```bash
   pip install -r requirements.txt

4. Retrain the model (optional):
   ```bash
   python train_model.py

5. Run the Streamlit app:
   ```bash
   streamlit run app.py

## 📂 Project Structure

titanic-survival-predictor/
│── train.csv               # Titanic dataset
│── train_model.py          # Model training, preprocessing, and saving
│── app.py                  # Streamlit app for predictions
│── model.pkl               # Trained SVM model
│── scaler.pkl              # StandardScaler object
│── columns.pkl             # Columns used in training
│── requirements.txt        # Project dependencies
│── README.md               # Project documentation



   
