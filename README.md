# **Defect Detection System Using Machine Learning**

## **Overview**
The **Defect Detection System** is a **Flask-based web application** that leverages **Machine Learning (ML)** techniques to automate defect classification. It utilizes **Logistic Regression** with **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance and improve predictive accuracy. The system enables users to **upload a dataset, train a model, and make predictions**, offering a seamless and efficient defect detection solution.

## **Key Features**
- **Dataset Upload:** Accepts CSV files for defect classification.
- **Data Preprocessing:** Handles missing values, normalizes features, and balances classes using SMOTE.
- **Model Training:** Implements **Logistic Regression** with **GridSearchCV** for hyperparameter tuning.
- **Model Persistence:** Saves trained models for future use.
- **Prediction System:** Allows users to upload new data for defect classification.
- **Performance Metrics:** Evaluates model performance with **accuracy and F1-score**.

## **Technology Stack**
- **Programming Language:** Python
- **Web Framework:** Flask
- **Data Processing:** Pandas, NumPy, Scikit-Learn, Imbalanced-learn
- **Model Storage:** Joblib
- **Frontend:** HTML, Jinja Templates

## **Installation & Setup**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/defect-detection.git
cd defect-detection
```

### **2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

## **Usage Guide**

### **1. Start the Flask Application**
```bash
python app.py
```
The application will be available at: **`http://127.0.0.1:5000/`**

### **2. Upload Dataset**
- Navigate to `/upload` in the browser.
- Upload a **CSV file** containing a `defectstatus` column.

### **3. Train the Model**
- The system preprocesses the data, applies **SMOTE**, and trains a **Logistic Regression** model.
- The trained model and scaler are saved as `trained_model.joblib` and `scaler.joblib`.

### **4. Make Predictions**
- Navigate to `/predict` and upload a **CSV file** (test data without labels).
- The system predicts defect status and presents the results.

## **Project Structure**
```bash
.
├── static/              # CSS, JS, and images
├── templates/           # HTML templates
│   ├── index.html       # Home Page
│   ├── train.html       # Model Training Page
│   ├── predict.html     # Prediction Page
│   ├── results.html     # Results Display
├── app.py               # Main Flask Application
├── requirements.txt     # Required Python Packages
├── README.md            # Project Documentation
```

## **API Endpoints**
| Endpoint    | Method | Description |
|------------|--------|-------------|
| `/`        | GET    | Home Page |
| `/upload`  | POST   | Upload Dataset |
| `/train`   | POST   | Train Machine Learning Model |
| `/predict` | POST   | Predict Defect Status |

## **Future Improvements**
- **Integration of Additional ML Models** (Random Forest, SVM, Deep Learning).
- **Deployment as a REST API** for integration with other systems.
- **Database Implementation** to store training history and logs.
- **Web-Based Interactive Dashboard** for improved user experience.



## **Author**
Mohammed Mujahid(https://github.com/mujahid027/)

