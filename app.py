import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import base64
from io import BytesIO

app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')
app.config['SECRET_KEY'] = os.urandom(24)

# Global variables
MODEL_PATH = 'app/models/random_forest_model.joblib'
SCALER_PATH = 'app/models/scaler.joblib'
DATA_PATH = 'agriculture_dataset.csv'

# Load data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load or train model
def get_model(retrain=False):
    if os.path.exists(MODEL_PATH) and not retrain:
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler
        except:
            pass
    
    # If model doesn't exist or retrain is True, train a new model
    return train_model()

def train_model():
    df = load_data()
    
    # Data preparation
    X, y, feature_names = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'rmse': round(rmse, 2),
        'r2': round(r2, 2),
        'feature_names': feature_names
    }
    
    with open('app/models/model_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return model, scaler

def preprocess_data(df):
    # Make a copy of the dataframe to avoid modifying the original
    df_encoded = df.copy()
    
    # Separate target variable
    y = df_encoded['Yield(tons)'].values
    
    # Process categorical columns
    categorical_columns = ['Farm_ID', 'Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']
    
    # Handle categorical features
    for col in categorical_columns:
        if col == 'Farm_ID':
            # Label encode Farm_ID
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
        else:
            # One-hot encode other categorical columns
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded_cols = ohe.fit_transform(df_encoded[[col]])
            encoded_cols_df = pd.DataFrame(
                encoded_cols,
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0][1:]],
                index=df_encoded.index
            )
            df_encoded = pd.concat([df_encoded, encoded_cols_df], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
    
    # Drop the target variable
    df_encoded.drop('Yield(tons)', axis=1, inplace=True)
    
    feature_names = df_encoded.columns.tolist()
    X = df_encoded.values
    
    return X, y, feature_names

def get_model_metrics():
    try:
        with open('app/models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return metrics
    except:
        return {'rmse': 0, 'r2': 0, 'feature_names': []}

# Generate plot for important features
def plot_feature_importance():
    model, _ = get_model()
    metrics = get_model_metrics()
    feature_names = metrics['feature_names']
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[-10:]  # Top 10 features
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='darkcyan', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    # Save to bytesIO object
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Encode to base64 for embedding in HTML
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    df = load_data()
    
    # Calculate summary statistics
    crop_counts = df['Crop_Type'].value_counts()
    soil_counts = df['Soil_Type'].value_counts()
    season_counts = df['Season'].value_counts()
    
    # Calculate average metrics by crop type
    crop_metrics = df.groupby('Crop_Type').agg({
        'Yield(tons)': 'mean',
        'Fertilizer_Used(tons)': 'mean',
        'Pesticide_Used(kg)': 'mean',
        'Water_Usage(cubic meters)': 'mean'
    }).reset_index()
    
    # Format columns
    for col in crop_metrics.columns:
        if col != 'Crop_Type':
            crop_metrics[col] = crop_metrics[col].round(2)
    
    # Model metrics
    model_metrics = get_model_metrics()
    
    # Generate feature importance plot
    feature_importance_plot = plot_feature_importance()
    
    return render_template(
        'dashboard.html', 
        crop_counts=crop_counts.to_dict(),
        soil_counts=soil_counts.to_dict(),
        season_counts=season_counts.to_dict(),
        crop_metrics=crop_metrics.to_dict(orient='records'),
        model_rmse=model_metrics['rmse'],
        model_r2=model_metrics['r2'],
        feature_importance_plot=feature_importance_plot
    )

@app.route('/data')
def data_view():
    df = load_data()
    return render_template('data.html', data=df.head(50).to_html(classes='table table-striped'))

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/api/plot/<plot_type>')
def get_plot(plot_type):
    df = load_data()
    
    if plot_type == 'crop_distribution':
        fig = px.pie(df, names='Crop_Type', title='Crop Type Distribution')
        return jsonify({'plot': fig.to_json()})
    
    elif plot_type == 'season_distribution':
        fig = px.pie(df, names='Season', title='Season Distribution')
        return jsonify({'plot': fig.to_json()})
        
    elif plot_type == 'yield_by_crop':
        fig = px.bar(df.groupby('Crop_Type')['Yield(tons)'].mean().reset_index(), 
                    x='Crop_Type', y='Yield(tons)', 
                    title='Average Yield by Crop Type')
        return jsonify({'plot': fig.to_json()})
        
    elif plot_type == 'yield_by_soil_type':
        fig = px.bar(df.groupby('Soil_Type')['Yield(tons)'].mean().reset_index(), 
                    x='Soil_Type', y='Yield(tons)', 
                    title='Average Yield by Soil Type')
        return jsonify({'plot': fig.to_json()})
        
    elif plot_type == 'yield_by_season':
        fig = px.bar(df.groupby('Season')['Yield(tons)'].mean().reset_index(), 
                    x='Season', y='Yield(tons)', 
                    title='Average Yield by Season')
        return jsonify({'plot': fig.to_json()})
        
    elif plot_type == 'yield_by_irrigation':
        fig = px.bar(df.groupby('Irrigation_Type')['Yield(tons)'].mean().reset_index(), 
                    x='Irrigation_Type', y='Yield(tons)', 
                    title='Average Yield by Irrigation Type')
        return jsonify({'plot': fig.to_json()})
        
    elif plot_type == 'correlation':
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                       title='Correlation Matrix')
        return jsonify({'plot': fig.to_json()})
        
    return jsonify({'error': 'Invalid plot type'})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    df = load_data()
    
    crop_types = df['Crop_Type'].unique().tolist()
    irrigation_types = df['Irrigation_Type'].unique().tolist()
    soil_types = df['Soil_Type'].unique().tolist()
    seasons = df['Season'].unique().tolist()
    
    prediction_result = None
    
    if request.method == 'POST':
        try:
            # Get user inputs
            farm_area = float(request.form.get('farm_area'))
            fertilizer_used = float(request.form.get('fertilizer_used'))
            pesticide_used = float(request.form.get('pesticide_used'))
            water_usage = float(request.form.get('water_usage'))
            crop_type = request.form.get('crop_type')
            irrigation_type = request.form.get('irrigation_type')
            soil_type = request.form.get('soil_type')
            season = request.form.get('season')
            
            # Create a sample dataframe with user inputs
            sample_data = pd.DataFrame({
                'Farm_ID': ['F999'],  # Dummy Farm_ID
                'Crop_Type': [crop_type],
                'Farm_Area(acres)': [farm_area],
                'Irrigation_Type': [irrigation_type],
                'Fertilizer_Used(tons)': [fertilizer_used],
                'Pesticide_Used(kg)': [pesticide_used],
                'Water_Usage(cubic meters)': [water_usage],
                'Soil_Type': [soil_type],
                'Season': [season],
                'Yield(tons)': [0]  # Dummy yield
            })
            
            # Preprocess the sample
            X_sample, _, _ = preprocess_data(pd.concat([df, sample_data], ignore_index=True))
            X_sample = X_sample[-1].reshape(1, -1)  # Get the last row which is our sample
            
            # Load model and scaler
            model, scaler = get_model()
            
            # Scale the sample
            X_sample = scaler.transform(X_sample)
            
            # Make prediction
            prediction = model.predict(X_sample)[0]
            prediction_result = round(prediction, 2)
            
        except Exception as e:
            prediction_result = f"Error: {str(e)}"
    
    return render_template(
        'predict.html',
        crop_types=crop_types,
        irrigation_types=irrigation_types,
        soil_types=soil_types,
        seasons=seasons,
        prediction=prediction_result
    )

@app.route('/retrain', methods=['GET'])
def retrain():
    # Retrain the model
    train_model()
    return redirect(url_for('dashboard'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Make sure model is trained before starting app
    get_model()
    app.run(debug=True)
