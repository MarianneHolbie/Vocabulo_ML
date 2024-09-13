import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from Vocabulo_quiz.src.db_connexion.database import get_db_connection


def prepare_training_data(conn):
    """
    Load data from the database using the provided connection.

    Parameters:
    conn (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.

    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    query = """
    SELECT 
        sq.score, 
        q.user_id,
        m.mot_id, m.niv_diff_id, m.frequence, m.gramm_id,
        c.name as category, 
        sc.name as subcategory,
        e.echelon_id,
        COALESCE(uwh.times_correct, 0) as times_correct,
        COALESCE(uwh.times_seen, 0) as times_seen,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - COALESCE(uwh.last_seen, '1970-01-01'::timestamp))) / 86400 as days_since_last_seen,
        diff.freqfilms, diff.freqlivres, diff.nbr_syll, diff.cp_cm2_sfi,
        q.date as quiz_date,
        COALESCE(em.scale, 'Bien') as user_feedback
    FROM score_quiz sq
    JOIN quiz q ON sq.quiz_id = q.quiz_id
    JOIN mot m ON sq.mot_id = m.mot_id
    JOIN mot_categorie mc ON m.mot_id = mc.mot_id
    JOIN categorie c ON mc.categorie_id = c.categorie_id
    LEFT JOIN mot_subcategory ms ON m.mot_id = ms.mot_id
    LEFT JOIN subcategory sc ON ms.subcat_id = sc.subcat_id
    LEFT JOIN user_word_history uwh ON q.user_id = uwh.user_id AND sq.mot_id = uwh.mot_id
    LEFT JOIN echelon_db e ON m.echelon_id = e.echelon_id
    LEFT JOIN diff_ortho diff ON m.mot_id = diff.mot_id
    LEFT JOIN eval_mot em ON q.quiz_id = em.quiz_id AND m.mot_id = em.mot_id
    """
    df = pd.read_sql(query, conn)
    print(f"Data loaded. DataFrame shape: {df.shape}")
    return df

def preprocess_data(df):
    """
    Preprocess the loaded data.

    This function performs the following steps:
    1. Processes temporal columns.
    2. Extracts temporal features.
    3. Encodes time cyclically.
    4. Defines feature types.
    5. Creates a preprocessor.
    6. Prepares features and target.

    Parameters:
    df (pd.DataFrame): DataFrame containing the loaded data.

    Returns:
    tuple: A tuple containing the features (X), target (y), and the preprocessor.
    """
    # Process temporal columns
    df['quiz_date'] = pd.to_datetime(df['quiz_date'])
    df['hour'] = df['quiz_date'].dt.hour
    df['day_of_week'] = df['quiz_date'].dt.dayofweek
    df['month'] = df['quiz_date'].dt.month

    # Encode time cyclically
    for col in ['hour', 'day_of_week', 'month']:
        max_val = 24 if col == 'hour' else 7 if col == 'day_of_week' else 12
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    # Define feature types
    numeric_features = ['frequence', 'times_correct', 'times_seen', 'days_since_last_seen',
                        'freqfilms', 'freqlivres', 'nbr_syll', 'cp_cm2_sfi']
    categorical_features = ['niv_diff_id', 'gramm_id', 'category', 'subcategory', 'echelon_id', 'user_feedback']
    cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('cyc', 'passthrough', cyclical_features)
        ])

    # Prepare features and target
    X = df.drop(['score', 'user_id', 'mot_id', 'quiz_date'], axis=1)
    y = df['score']

    return X, y, preprocessor

def prepare_data_for_model():
    """
    Prepare data for the hybrid recommender model.

    This function performs the following steps:
    1. Connects to the database.
    2. Loads data from the database.
    3. Preprocesses the data.
    4. Separates features and target.

    Returns:
    tuple: A tuple containing the features (X), target (y), and the preprocessor.
    """
    conn = get_db_connection()
    df = prepare_training_data(conn)
    X, y, preprocessor = preprocess_data(df)
    return X, y, preprocessor

if __name__ == "__main__":
    X, y, preprocessor = prepare_data_for_model()
    print("Data preparation complete.")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")