import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ndcg_score, confusion_matrix, average_precision_score
import lightgbm as lgb
import catboost as cb
import optuna
import os
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import warnings
from Vocabulo_quiz.src.db_connexion.database import get_db_connection

warnings.filterwarnings("ignore", category=FutureWarning)



def prepare_data(conn):
    """Fetches and prepares the data from the database.

    Args:
        conn: Database connection object.

    Returns:
        df: Prepared data as a DataFrame.
    """
    query = """
        SELECT 
        sq.score, 
        q.user_id,
        m.mot_id, m.mot, 
        COALESCE(m.niv_diff_id, 0) as niv_diff_id, 
        COALESCE(m.frequence, 0) as frequence, 
        COALESCE(m.gramm_id, 0) as gramm_id,
        CASE 
            WHEN c.name = 'basique' OR sc.categorie_id = (SELECT categorie_id FROM categorie WHERE name = 'basique') THEN 'basique'
            WHEN c.name IS NOT NULL THEN c.name
            ELSE 'non_categorise'
        END as category,
        COALESCE(sc.name, 'sans_sous_categorie') as subcategory,
        COALESCE(e.echelon_id, 0) as echelon_id,
        COALESCE(uwh.times_correct, 0) as times_correct,
        COALESCE(uwh.times_seen, 0) as times_seen,
        COALESCE(uwh.last_seen, '1970-01-01'::timestamp) as last_seen,
        COALESCE(diff.freqfilms, 0) as freqfilms, 
        COALESCE(diff.freqlivres, 0) as freqlivres, 
        COALESCE(diff.nbr_syll, 0) as nbr_syll, 
        COALESCE(diff.cp_cm2_sfi, 0) as cp_cm2_sfi,
        COALESCE(q.date, '1970-01-01'::timestamp) as quiz_date
    FROM score_quiz sq
    JOIN quiz q ON sq.quiz_id = q.quiz_id
    JOIN mot m ON sq.mot_id = m.mot_id
    LEFT JOIN mot_categorie mc ON m.mot_id = mc.mot_id
    LEFT JOIN categorie c ON mc.categorie_id = c.categorie_id
    LEFT JOIN mot_subcategory ms ON m.mot_id = ms.mot_id
    LEFT JOIN subcategory sc ON ms.subcat_id = sc.subcat_id
    LEFT JOIN user_word_history uwh ON q.user_id = uwh.user_id AND sq.mot_id = uwh.mot_id
    LEFT JOIN echelon_db e ON m.echelon_id = e.echelon_id
    LEFT JOIN diff_ortho diff ON m.mot_id = diff.mot_id
    """
    df = pd.read_sql(query, conn)
    return df


def preprocess_data(df):
    """Preprocesses the input DataFrame and prepares features for model training.

    Args:
        df: Input DataFrame containing raw data.

    Returns:
        X: Feature matrix.
        y: Target vector.
    """
    # Processing time columns
    df['quiz_date'] = pd.to_datetime(df['quiz_date'])
    df['last_seen'] = pd.to_datetime(df['last_seen'])
    df['days_since_last_seen'] = (df['quiz_date'] - df['last_seen']).dt.total_seconds() / 86400

    # Extraction of temporal characteristics
    df['hour'] = df['quiz_date'].dt.hour
    df['day_of_week'] = df['quiz_date'].dt.dayofweek
    df['month'] = df['quiz_date'].dt.month

    # Cyclic time encoding
    for col in ['hour', 'day_of_week', 'month']:
        max_val = 24 if col == 'hour' else 7 if col == 'day_of_week' else 12
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    # Word Embedding - Optimised to prevent fragmentation
    sentences = df['mot'].apply(lambda x: x.split()).tolist()
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word_embeddings = df['mot'].apply(lambda x: word2vec_model.wv[x] if x in word2vec_model.wv else np.zeros(100))
    word_embedding_cols = [f'embed_{i}' for i in range(100)]
    word_embedding_df = pd.DataFrame(word_embeddings.tolist(), columns=word_embedding_cols)
    df = pd.concat([df, word_embedding_df], axis=1)

    # Definition of columns
    numeric_features = ['frequence', 'times_correct', 'times_seen', 'freqfilms', 'freqlivres', 'nbr_syll', 'cp_cm2_sfi',
                        'days_since_last_seen']
    categorical_features = ['niv_diff_id', 'gramm_id', 'category', 'subcategory', 'echelon_id']
    cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
    embedding_features = word_embedding_cols

    # Preparing features and targets
    feature_columns = numeric_features + categorical_features + cyclical_features + embedding_features
    X = df[['user_id', 'mot_id'] + feature_columns]
    y = df['score']

    return X, y


def get_user_level(conn, user_id):
    query = """
    SELECT 
        COALESCE(AVG(CASE WHEN sq.score THEN 1 ELSE 0 END), 0) as avg_score,
        COUNT(DISTINCT CASE WHEN c.name = 'basique' THEN m.mot_id END) as basic_words_seen,
        COUNT(DISTINCT CASE WHEN c.name = 'basique' AND sq.score THEN m.mot_id END) as basic_words_correct,
        COUNT(DISTINCT m.mot_id) as total_words_seen,
        MAX(e.echelon_id) as max_echelon
    FROM score_quiz sq
    JOIN quiz q ON sq.quiz_id = q.quiz_id
    JOIN mot m ON sq.mot_id = m.mot_id
    JOIN mot_categorie mc ON m.mot_id = mc.mot_id
    JOIN categorie c ON mc.categorie_id = c.categorie_id
    LEFT JOIN echelon_db e ON m.echelon_id = e.echelon_id
    WHERE q.user_id = %s
    """
    df = pd.read_sql(query, conn, params=[user_id])
    avg_score = df['avg_score'].iloc[0]
    basic_words_seen = df['basic_words_seen'].iloc[0]
    basic_words_correct = df['basic_words_correct'].iloc[0]
    total_words_seen = df['total_words_seen'].iloc[0]
    max_echelon = df['max_echelon'].iloc[0]

    basic_mastery = basic_words_correct / basic_words_seen if basic_words_seen > 0 else 0
    level = min(5, max(1, int(avg_score * 5) + 1))  # Niveau de 1 à 5

    return level, basic_mastery, total_words_seen, max_echelon

def create_preprocessor():
    """
    Creates a preprocessor for the data.

    Returns:
        preprocessor: A preprocessor object to transform the data.
    """
    numeric_features = ['frequence', 'times_correct', 'times_seen', 'freqfilms', 'freqlivres', 'nbr_syll', 'cp_cm2_sfi', 'days_since_last_seen']
    categorical_features = ['niv_diff_id', 'gramm_id', 'category', 'subcategory', 'echelon_id']
    cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
    embedding_features = [f'embed_{i}' for i in range(100)]

    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('cyc', 'passthrough', cyclical_features),
            ('emb', 'passthrough', embedding_features)
        ])

    return preprocessor, numeric_features, categorical_features, cyclical_features, embedding_features


def objective(trial, X, y, model_type):
    """
    Objective function for hyperparameter optimization using Optuna.

    This function defines the search space for hyperparameters and evaluates the model performance
    using cross-validation.

    Parameters:
    trial (optuna.trial.Trial): A trial object that suggests hyperparameters.
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target vector.
    model_type (str): Type of model to be optimized ('rf' for RandomForest, 'lgb' for LightGBM, 'cb' for CatBoost).

    Returns:
    float: Mean F1 score from cross-validation.
    """
    preprocessor, _, _, _, _ = create_preprocessor()

    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight='balanced')
    elif model_type == 'lgb':
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        }
        model = lgb.LGBMClassifier(**params, random_state=42, class_weight='balanced')
    elif model_type == 'cb':
        params = {
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'iterations': trial.suggest_int('iterations', 100, 1000),
        }
        model = cb.CatBoostClassifier(**params, random_state=42, verbose=0)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    return scores.mean()


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='rf'):
    """
    Trains and evaluates a machine learning model.

    Args:
        X_train: Feature matrix for training.
        y_train: Target vector for training.
        X_test: Feature matrix for testing.
        y_test: Target vector for testing.
        model_type: Type of model to be trained ('rf', 'lgb', 'cb').

    Returns:
        model: Trained model object.
    """
    preprocessor, numeric_features, categorical_features, cyclical_features, embedding_features = create_preprocessor()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, model_type), n_trials=100)

    best_params = study.best_params
    print(f"Best hyperparameters for {model_type}:", best_params)

    if model_type == 'rf':
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1, class_weight='balanced')
    elif model_type == 'lgb':
        best_model = lgb.LGBMClassifier(**best_params, random_state=42, class_weight='balanced')
    elif model_type == 'cb':
        best_model = cb.CatBoostClassifier(**best_params, random_state=42, verbose=0)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'model_type': model_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'ndcg': ndcg_score([y_test], [y_pred_proba])
    }

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame([metrics])
    if os.path.exists('../../reports/performance_metrics/model2_metrics.csv'):
        metrics_df.to_csv('model2_metrics.csv', mode='a', header=False, index=False)
    else:
        metrics_df.to_csv('model2_metrics.csv', mode='w', header=True, index=False)

    # Feature importance
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        feature_importance = pipeline.named_steps['classifier'].feature_importances_

        # Get feature names
        feature_names = (
                numeric_features +
                pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
                    'onehot'].get_feature_names_out(categorical_features).tolist() +
                cyclical_features +
                embedding_features
        )

        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(importance_df.head(10))
    else:
        print("\nWarning: This model doesn't support feature importance.")

    return pipeline


def get_word_details(conn, word_ids):
    """
    Fetches details of words from the database based on provided word IDs.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): The database connection to execute the query.
    word_ids (list): List of word IDs to fetch details for.

    Returns:
    dict: A dictionary where keys are word IDs and values are dictionaries containing word details.
    """
    query = """
    SELECT m.mot_id, m.mot, m.niv_diff_id, c.name as category, sc.name as subcategory
    FROM mot m
    LEFT JOIN mot_categorie mc ON m.mot_id = mc.mot_id
    LEFT JOIN categorie c ON mc.categorie_id = c.categorie_id
    LEFT JOIN mot_subcategory ms ON m.mot_id = ms.mot_id
    LEFT JOIN subcategory sc ON ms.subcat_id = sc.subcat_id
    WHERE m.mot_id IN %(word_ids)s
    """
    df = pd.read_sql(query, conn, params={'word_ids': tuple(word_ids)})
    return {row['mot_id']: row.to_dict() for _, row in df.iterrows()}

def predict_recommendations(model, X_test, user_mapping, word_mapping, conn, num_recommendations=5):
    """
    Generates recommendations for a test user.

    Args:
        model: Trained model object.
        X_test: Feature matrix for testing.
        user_mapping: Dictionary mapping user IDs to indices.
        word_mapping: Dictionary mapping word IDs to indices.
        conn: Database connection object.

    Returns:
        recommendations: Dictionary containing recommendations for the test user.
    """
    user_ids = X_test['user_id'].unique()
    recommendations = {}

    for user_id in user_ids:
        user_data = X_test[X_test['user_id'] == user_id]
        user_features = user_data.drop(['user_id', 'mot_id'], axis=1)

        predictions = model.predict_proba(user_features)[:, 1]

        user_level, basic_mastery, total_words_seen, max_echelon = get_user_level(conn, user_id)

        # Adjust predictions according to user level
        adjusted_predictions = adjust_predictions(predictions, user_level, basic_mastery, user_data)

        top_indices = np.argsort(adjusted_predictions)[-num_recommendations:]

        recommended_word_ids = user_data.iloc[top_indices]['mot_id'].tolist()

        word_details = get_word_details(conn, recommended_word_ids)

        recommendations[user_id] = [
            {
                'mot_id': int(word_id),
                'mot': word_details[word_id]['mot'],
                'niveau_difficulte': word_details[word_id]['niv_diff_id'],
                'categorie': word_details[word_id]['category'],
                'sous_categorie': word_details[word_id]['subcategory']
            }
            for word_id in recommended_word_ids
        ]

    return recommendations


def adjust_predictions(predictions, user_level, basic_mastery, user_data):
    """
    Adjusts the prediction scores based on user level, basic mastery, and word difficulty.

    Parameters:
    predictions (np.array): Array of prediction scores.
    user_level (int): The level of the user.
    basic_mastery (float): The basic mastery score of the user.
    user_data (pd.DataFrame): DataFrame containing user and word details.

    Returns:
    np.array: Adjusted prediction scores.
    """
    # Define basic categories
    basic_categories = ['basique', 'Hiver', 'Noël', 'conte', 'corps', 'jouet', 'lieux', 'été',
                        'bébé', 'cirque', 'fleurs', 'maison', 'école', 'Automne', 'Météo', 'animaux', 'cuisine',
                        'famille', 'musique', 'nombres', 'piscine', 'couleurs','métiers', 'position', 'Halloween',
                        'personnes', 'printemps', 'émotions', 'nourriture', 'transports', 'vêtements',
                        'Verbes d\'action', 'sports et loisirs', 'fruits et légumes', 'formes géométriques']

    adjusted_predictions = predictions.copy()

    for i, (_, word) in enumerate(user_data.iterrows()):
        # Increase the score for basic words if basic proficiency is low
        if word['category'] in basic_categories and basic_mastery < 0.7:
            adjusted_predictions[i] *= 1.5

        # Adjust according to word difficulty and user level
        word_difficulty = word['niv_diff_id']
        if word_difficulty <= user_level:
            adjusted_predictions[i] *= 1.2
        else:
            adjusted_predictions[i] *= 0.8

    return adjusted_predictions



if __name__ == "__main__":
    conn = get_db_connection()
    df = prepare_data(conn)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessor()

    # Create user and word mappings
    user_mapping = {user: i for i, user in enumerate(X['user_id'].unique())}
    word_mapping = {word: i for i, word in enumerate(X['mot_id'].unique())}

    # Training and evaluating models
    models = {}
    for model_type in ['rf', 'lgb', 'cb']:
        print(f"\nTraining and assessment of the model {model_type}")
        models[model_type] = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type)

    # Choosing the best model
    best_model_type = max(models, key=lambda k: f1_score(y_test, models[k].predict(X_test)))
    best_model = models[best_model_type]
    print(f"\nBest model : {best_model_type}")

    # Generate recommendations for a test user
    test_user_id = X_test['user_id'].iloc[0]  # Take the first test user as an example
    recommendations = predict_recommendations(best_model, X_test, user_mapping, word_mapping, conn)
    print(f"\nRecommendations for users {test_user_id}:")
    print(recommendations[test_user_id])

    conn.close()