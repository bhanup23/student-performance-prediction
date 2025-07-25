import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Expanded_data_with_more_features.csv')
    return df

# Feature Engineering and Preprocessing
def preprocess_data(df):
    df['LunchType'] = df['LunchType'].fillna(df['LunchType'].mode()[0])
    df['TestPrep'] = df['TestPrep'].fillna('none')
    df['ParentEduc'] = df['ParentEduc'].fillna(df['ParentEduc'].mode()[0])
    df['ParentMaritalStatus'] = df['ParentMaritalStatus'].fillna(df['ParentMaritalStatus'].mode()[0])
    df['PracticeSport'] = df['PracticeSport'].fillna(df['PracticeSport'].mode()[0])
    df['IsFirstChild'] = df['IsFirstChild'].fillna(df['IsFirstChild'].mode()[0])
    df['TransportMeans'] = df['TransportMeans'].fillna(df['TransportMeans'].mode()[0])
    df['WklyStudyHours'] = df['WklyStudyHours'].fillna(df['WklyStudyHours'].mode()[0])
    
    # Create new feature
    df['AvgScore'] = (df['MathScore'] + df['ReadingScore'] + df['WritingScore']) / 3
    
    # Encode categorical variables
    categorical_cols = ['Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
                        'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild',
                        'TransportMeans', 'WklyStudyHours']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop target leakage
    df_encoded = df_encoded.drop(['ReadingScore', 'WritingScore'], axis=1)

    # Drop rows with any remaining NaNs (for safety)
    df_encoded = df_encoded.dropna()

    return df_encoded

# Model training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, X_train.columns, mse, r2, X_test, y_test, y_pred

# Plotting function
def plot_feature_importance(model, columns):
    importance = np.abs(model.coef_)
    feature_importance = pd.Series(importance, index=columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
    plt.title('Top 10 Feature Importance for Math Score Prediction')
    plt.xlabel('Coefficient Magnitude')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png')
    plt.close()

# Streamlit interface
def main():
    st.title('Student Performance Prediction Dashboard')
    st.write('Predict student math scores using linear regression and explore data insights.')

    df = load_data()
    df_encoded = preprocess_data(df)

    X = df_encoded.drop('MathScore', axis=1)
    y = df_encoded['MathScore']

    model, scaler, columns, mse, r2, X_test, y_test, y_pred = train_model(X, y)

    analysis_type = st.sidebar.selectbox('Select Analysis', ['Model Results', 'Feature Importance', 'Prediction'])

    if analysis_type == 'Model Results':
        st.header('Model Performance')
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'R² Score: {r2:.2f}')
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Math Score')
        plt.ylabel('Predicted Math Score')
        plt.title('Actual vs Predicted Math Scores')
        plt.savefig('actual_vs_predicted.png')
        st.image('actual_vs_predicted.png')

    elif analysis_type == 'Feature Importance':
        st.header('Feature Importance')
        plot_feature_importance(model, columns)
        st.image('feature_importance.png')
        st.write('Shows the top 10 features influencing math score predictions.')

    elif analysis_type == 'Prediction':
        st.header('Predict Math Score')
        gender = st.selectbox('Gender', ['male', 'female'])
        ethnic_group = st.selectbox('Ethnic Group', ['group A', 'group B', 'group C', 'group D', 'group E'])
        parent_educ = st.selectbox('Parental Education', ['some high school', 'high school', 'some college', 
                                                           'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'])
        lunch_type = st.selectbox('Lunch Type', ['standard', 'free/reduced'])
        test_prep = st.selectbox('Test Preparation', ['none', 'completed'])
        study_hours = st.selectbox('Weekly Study Hours', ['< 5', '5 - 10', '> 10'])

        input_data = pd.DataFrame({
            'Gender': [gender],
            'EthnicGroup': [ethnic_group],
            'ParentEduc': [parent_educ],
            'LunchType': [lunch_type],
            'TestPrep': [test_prep],
            'WklyStudyHours': [study_hours],
            'AvgScore': [0]  # Placeholder
        })

        input_encoded = pd.get_dummies(input_data, columns=['Gender', 'EthnicGroup', 'ParentEduc', 
                                                            'LunchType', 'TestPrep', 'WklyStudyHours'])
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]

        st.write(f'Predicted Math Score: {prediction:.2f}')

if __name__ == '__main__':
    main()
