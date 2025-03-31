import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare_data(n_samples=5000):
    np.random.seed(42)
    
    # Example data
    df = pd.DataFrame({
        'premium': np.random.uniform(200, 1000, n_samples),
        'competitor_price': np.random.uniform(180, 950, n_samples),
        'vehicle_type': np.random.choice(['Hatchback', 'SUV', 'Sedan'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'location_risk': np.random.uniform(0.5, 3.0, n_samples),
        'economic_index': np.random.uniform(90, 110, n_samples)
    })

    # Handle categorical variables (e.g., vehicle_type)
    df = pd.get_dummies(df, columns=['vehicle_type'], drop_first=True)

    # Add target variable
    df['log_premium'] = np.log(df['premium'])

    # Handle NaNs or infinities if they exist
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Features and target
    X = df.drop(columns=['premium', 'log_premium'])

    # Check for non-numeric columns and remove them (for debugging)
    X = X.select_dtypes(include=[np.number])

    # Ensure that target is numeric
    y = df['log_premium']

    # Split into train and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)
