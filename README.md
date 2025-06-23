# student-performance-prediction
Student Performance Prediction
Overview
This project predicts student math scores using linear regression on the "Students Performance in Exams" Kaggle dataset. It includes feature engineering, model evaluation, and an interactive Streamlit dashboard to visualize results and make predictions based on user input.
Features

Data Preprocessing: Handles missing values and encodes categorical variables.
Feature Engineering: Creates an average score feature and applies one-hot encoding.
Linear Regression: Predicts math scores with evaluation metrics (MSE, R²).
Visualizations: Feature importance and actual vs. predicted score plots.
Streamlit Dashboard: Interactive UI for model results, feature analysis, and score prediction.

Dataset

Source: Students Performance in Exams by Desalegn Butter on Kaggle.
File: Expanded_data_with_more_features.csv
Details: Contains ~30,000 records with features like gender, ethnicity, parental education, lunch type, test preparation, and math/reading/writing scores.

Requirements

Python 3.8+
Libraries: Install via pip install -r requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit



Setup Instructions

Clone the repository:git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction


Install dependencies:pip install -r requirements.txt


Download the dataset from Kaggle and place Expanded_data_with_more_features.csv in the project directory.
Run the Streamlit app:streamlit run student_performance_prediction.py


Open the provided URL (e.g., http://localhost:8501) in your browser.

Deployment

Streamlit Community Cloud: Deploy for a live demo:
Push the project to GitHub.
Connect to Streamlit Community Cloud and select the repository.
Ensure requirements.txt and the dataset are included.


Live Demo: (Add your Streamlit URL here, e.g., https://your-student-performance.streamlit.app).

Project Structure

student_performance_prediction.py: Main script for analysis, model training, and Streamlit app.
Expanded_data_with_more_features.csv: Dataset file (download from Kaggle).
requirements.txt: Python dependencies.
feature_importance.png, actual_vs_predicted.png: Generated visualization files.

Results

Model Performance: Achieves ~88% R² score (based on typical linear regression results for this dataset).
Key Features: Parental education, lunch type, and test preparation significantly influence math scores.
Insights: Visualizations highlight feature importance and model accuracy.

Future Improvements

Explore advanced models (e.g., Random Forest, XGBoost).
Add clustering (e.g., K-means) to group students by performance.
Incorporate more feature engineering (e.g., interaction terms).

Author

Your Name
GitHub: yourusername
Email: your.email@example.com

License
MIT License
