import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import optuna
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import json
from Vocabulo_quiz.src.db_connexion.database import get_db_connection

OUTPUT_DIR = '/app/output_FINAL'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration loggin
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_training_data(conn):
    """
    Prepares the training data by executing a SQL query and returning the result as a DataFrame.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.

    Returns:
    pd.DataFrame: DataFrame containing the prepared training data.
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
    return df


def preprocess_data(df):
    """
    Preprocesses the data by handling temporal columns, extracting features, and applying transformations.

    Parameters:
    df (pd.DataFrame): DataFrame containing the raw data.

    Returns:
    tuple: Transformed features (X), target variable (y), feature names, and preprocessor object.
    """
    # Processing time columns
    df['quiz_date'] = pd.to_datetime(df['quiz_date'])
    df['hour'] = df['quiz_date'].dt.hour
    df['day_of_week'] = df['quiz_date'].dt.dayofweek
    df['month'] = df['quiz_date'].dt.month

    #  Cyclic time encoding
    for col in ['hour', 'day_of_week', 'month']:
        max_val = 24 if col == 'hour' else 7 if col == 'day_of_week' else 12
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    # Defining feature types
    numeric_features = ['frequence', 'times_correct', 'times_seen', 'days_since_last_seen',
                        'freqfilms', 'freqlivres', 'nbr_syll', 'cp_cm2_sfi']
    categorical_features = ['niv_diff_id', 'gramm_id', 'category', 'subcategory', 'echelon_id', 'user_feedback']
    cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']

    # Creating the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('cyc', 'passthrough', cyclical_features)
        ])

    #  Preparing features and targets
    X = df.drop(['score', 'user_id', 'mot_id', 'quiz_date'], axis=1)
    y = df['score']

    return X, y, preprocessor


def objective(trial, X, y, preprocessor):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    trial (optuna.trial.Trial): Optuna trial object.
    X (np.array): Feature matrix.
    y (np.array): Target variable.
    preprocessor (sklearn.compose.ColumnTransformer): Preprocessor object used for data transformation.

    Returns:
    float: F1 score of the model.
    """
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
    }

    model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    return scores.mean()


def train_model(X, y, preprocessor):
    """
    Trains a machine learning model using Optuna for hyperparameter optimization.

    Parameters:
    X (np.array): Feature matrix.
    y (np.array): Target variable.
    preprocessor (sklearn.compose.ColumnTransformer): Preprocessor object used for data transformation.

    Returns:
    sklearn.pipeline.Pipeline: Trained machine learning pipeline.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=200)

    best_params = study.best_params
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X, y)

    # Save the entire pipeline
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, 'xgboost_pipeline.joblib'))

    return pipeline, study


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluates the trained model on the test data.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    X_test (np.array): Testing feature matrix.
    y_test (np.array): Testing target variable.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred_proba),
        "Log Loss": log_loss(y_test, y_pred_proba)
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # save pipeline
    joblib.dump(pipeline, 'recommendation_pipeline.joblib')

    return metrics


def plot_feature_importance(pipeline, feature_names):
    """
    Plots the feature importance of the trained model.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    feature_names (list): List of feature names.
    """
    model = pipeline.named_steps['classifier']
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(pos, importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(feature_names)[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance (MDI)')
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output_final')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()


def interpret_model(pipeline, X, feature_names):
    """
    Interprets the trained model using SHAP values and plots the summary.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    X (np.array): Feature matrix.
    feature_names (list): List of feature names.
    """
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(pipeline.named_steps['preprocessor'].transform(X))

    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, pipeline.named_steps['preprocessor'].transform(X), feature_names=feature_names,
                      plot_type="bar")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()


def save_model(pipeline, metrics, study):
    """
    Saves the trained model, metrics, and Optuna study to disk.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    metrics (dict): Dictionary containing evaluation metrics.
    study (optuna.study.Study): Optuna study object.
    """
    joblib.dump(pipeline, 'xgboost_pipeline.joblib')

    with open('model_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

    joblib.dump(study, 'optuna_study.joblib')


def save_results(pipeline, metrics, study):
    """
    Saves the results of the model training and evaluation to disk.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    metrics (dict): Dictionary containing evaluation metrics.
    study (optuna.study.Study): Optuna study object.
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'output_final')
    os.makedirs(output_dir, exist_ok=True)

    try:
        joblib.dump(pipeline['preprocessor'], os.path.join(output_dir, 'preprocessor.joblib'))
        joblib.dump(pipeline['classifier'], os.path.join(output_dir, 'classifier.joblib'))
        logging.info(f"Preprocessor and classifier saved in{output_dir}")
    except Exception as e:
        logging.error(f"Backup error : {str(e)}")

    try:
        with open(os.path.join(output_dir, 'model_metrics_final.txt'), 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        logging.info(f"Metrics saved in {output_dir}")
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")

    try:
        joblib.dump(study, os.path.join(output_dir, 'optuna_study_final.joblib'))
        logging.info(f"Optuna study saved in {output_dir}")
    except Exception as e:
        logging.error(f"Error saving the Optuna study : {str(e)}")

    logging.info(f"All results have been saved in {output_dir}")


def load_model():
    """
    Loads the trained model from disk.

    Returns:
    sklearn.pipeline.Pipeline: Loaded machine learning pipeline.
    """
    pipeline = joblib.load('xgboost_pipeline.joblib')
    return pipeline


def calculate_forgetting_curve(strength, time):
    """
    Calculates the forgetting curve based on the strength and time.

    Parameters:
    strength (float): Strength of the memory.
    time (float): Time elapsed.

    Returns:
    float: Forgetting curve value.
    """
    return np.exp(-time / strength)


def calculate_optimal_review_time(strength, threshold=0.5):
    """
    Calculates the optimal review time based on the strength and threshold.

    Parameters:
    strength (float): Strength of the memory.
    threshold (float): Threshold for the forgetting curve.

    Returns:
    float: Optimal review time.
    """
    return -strength * np.log(threshold)


def get_user_feedback(conn, user_id):
    """
    Retrieves user feedback from the database.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.
    user_id (int): User ID.

    Returns:
    dict: Dictionary containing user feedback.
    """
    query = """
    SELECT m.mot_id, em.scale
    FROM eval_mot em
    JOIN quiz q ON em.quiz_id = q.quiz_id
    JOIN mot m ON em.mot_id = m.mot_id
    WHERE q.user_id = %s
    """
    return pd.read_sql(query, conn, params=(user_id,)).set_index('mot_id')['scale'].to_dict()


def get_word_recommendations(conn, user_id, pipeline, num_words=10):
    """
    Generates word recommendations for a user based on the trained model.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.
    user_id (int): User ID.
    pipeline (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    num_words (int): Number of words to recommend.

    Returns:
    pd.DataFrame: DataFrame containing recommended words.
    """
    user_level, basic_mastery, total_words_seen, max_echelon, is_new_user = get_user_level(conn, user_id)
    user_feedback = get_user_feedback(conn, user_id)
    logging.info(f"User level: {user_level}, Basic mastery: {basic_mastery}, Total words seen: {total_words_seen}, Is new user: {is_new_user}")

    query = """
        SELECT m.mot_id, m.mot, m.niv_diff_id, COALESCE(m.frequence, 0) as frequence, m.gramm_id, ls.url_def, ls.url_sign, 
            c."name" as category,
            sc."name" as subcategory,
            e.echelon_id,
            COALESCE(diff.freqfilms, 0) as freqfilms, 
            COALESCE(diff.freqlivres, 0) as freqlivres, 
            COALESCE(diff.nbr_syll, 0) as nbr_syll, 
            COALESCE(diff.cp_cm2_sfi, 0) as cp_cm2_sfi,
            COALESCE(uwh.times_correct, 0) as times_correct,
            COALESCE(uwh.times_seen, 0) as times_seen,
            COALESCE(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - uwh.last_seen)) / 86400, 9999) as days_since_last_seen,
            CURRENT_TIMESTAMP as current_time
        FROM mot m
        LEFT JOIN lsf_signe ls ON m.mot_id = ls.mot_id
        LEFT JOIN mot_categorie mc ON m.mot_id = mc.mot_id
        LEFT JOIN categorie c ON mc.categorie_id = c.categorie_id
        LEFT JOIN mot_subcategory ms ON m.mot_id = ms.mot_id
        LEFT JOIN subcategory sc ON ms.subcat_id = sc.subcat_id
        LEFT JOIN echelon_db e ON m.echelon_id = e.echelon_id
        LEFT JOIN diff_ortho diff ON m.mot_id = diff.mot_id
        LEFT JOIN user_word_history uwh ON uwh.user_id = %s AND uwh.mot_id = m.mot_id
        WHERE ls.url_sign != 'Non spécifié' OR ls.url_def != 'Non spécifié'
        """
    df = pd.read_sql(query, conn, params=(user_id,))
    logging.info(f"Retrieved {len(df)} words from database")
    logging.info(f"Nombre total de mots récupérés : {len(df)}")
    logging.info(f"Nombre de mots basiques : {len(df[df['category'] == 'basique'])}")
    logging.info(f"Catégories uniques : {df['category'].unique()}")

    # Prétraitement des données
    df['hour'] = pd.to_datetime(df['current_time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['current_time']).dt.dayofweek
    df['month'] = pd.to_datetime(df['current_time']).dt.month

    for col in ['hour', 'day_of_week', 'month']:
        max_val = 24 if col == 'hour' else 7 if col == 'day_of_week' else 12
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    # Ajuster pour les nouveaux utilisateurs
    if is_new_user:
        df['times_seen'] = 0
        df['times_correct'] = 0
        df['days_since_last_seen'] = 9999  # Une grande valeur pour indiquer qu'ils n'ont jamais vu le mot

    # Ajouter une colonne de feedback fictif (puisque c'est une prédiction)
    df['user_feedback'] = 'Bien'

    X_pred = df.drop(['mot_id', 'mot', 'current_time', 'url_sign', 'url_def'], axis=1)

    # Prédire la probabilité de réponse correcte
    probabilities = pipeline.predict_proba(X_pred)[:, 1]

    df['prediction'] = probabilities
    df['difficulty_score'] = df['niv_diff_id'] / 5  # Normaliser la difficulté
    df['recency_score'] = 1 / (df['days_since_last_seen'] + 1)
    df['novelty_score'] = 1 / (df['times_seen'] + 1)

    df['recommendation_score'] = df.apply(
        lambda row: calculate_recommendation_score(row, user_level, basic_mastery, total_words_seen, user_feedback,
                                                   is_new_user),
        axis=1
    )

    # Sélection des mots recommandés
    recommended_words = select_diverse_words(df, num_words, basic_mastery, is_new_user)
    logging.info(f"Selected {len(recommended_words)} words for recommendation")

    logging.info(
        f"Niveau de l'utilisateur: {user_level}, Maîtrise de base: {basic_mastery:.2f}, Mots vus: {total_words_seen}")
    logging.info(f"Nombre de mots recommandés: {len(recommended_words)}")
    logging.info(f"Distribution des niveaux de difficulté: {recommended_words['niv_diff_id'].value_counts().to_dict()}")
    logging.info(f"Top 10 mots recommandés avant sélection finale:")
    logging.info(df.nlargest(10, 'recommendation_score')[['mot', 'category', 'niv_diff_id', 'recommendation_score']])

    return recommended_words[['mot_id', 'mot', 'category', 'subcategory', 'niv_diff_id',
                              'prediction', 'difficulty_score', 'recency_score', 'novelty_score',
                              'recommendation_score']]


def calculate_recommendation_score(row, user_level, basic_mastery, total_words_seen, user_feedback, is_new_user):
    """
    Calculates the recommendation score for a word.

    Parameters:
    row (pd.Series): Row of the DataFrame containing word data.
    user_level (int): User level.
    basic_mastery (float): Basic mastery level.
    total_words_seen (int): Total number of words seen by the user.
    user_feedback (dict): Dictionary containing user feedback.
    is_new_user (bool): Whether the user is new.

    Returns:
    float: Recommendation score.
    """
    prediction = row['prediction']
    difficulty = row['niv_diff_id']
    times_seen = row['times_seen']
    days_since_last_seen = row['days_since_last_seen']
    is_basic = row['category'] == 'basique'
    frequence = row['frequence']

    # Facteur de difficulté
    difficulty_factor = 1 - abs(difficulty - user_level) / max(user_level, 5)

    # Facteur de nouveauté
    novelty = 1 / (times_seen + 1)

    # Facteur de récence
    recency = 1 / (days_since_last_seen + 1)

    # Score de difficulté basé sur la fréquence
    difficulty_score = 1 - (frequence / 100) if frequence is not None and frequence != 0 else 0.5

    # Bonus pour les mots basiques, ajusté selon la maîtrise de base
    basic_bonus = 1.5 + (1 - basic_mastery) if is_basic else 1

    # Ajustement basé sur le feedback utilisateur (si disponible)
    feedback_factor = 1
    if row['mot_id'] in user_feedback:
        if user_feedback[row['mot_id']] == 'trop facile':
            feedback_factor = 0.8
        elif user_feedback[row['mot_id']] == 'trop difficile':
            feedback_factor = 1.2

    # Facteur d'expérience de l'utilisateur
    experience_factor = 1 / (total_words_seen + 1)

    prediction_factor = 1 - prediction

    score = (
        difficulty_factor * 0.2 +
        novelty * 0.2 +
        recency * 0.1 +
        difficulty_score * 0.2 +
        prediction_factor * 0.2 +
        experience_factor * 0.1
    ) * basic_bonus * feedback_factor

    if is_new_user:
        if is_basic:
            score *= 1.5
        if difficulty <= 2:
            score *= 1.3

    return score


def select_diverse_words(df, num_words, basic_mastery, is_new_user):
    """
    Selects a diverse set of words for recommendation.

    Parameters:
    df (pd.DataFrame): DataFrame containing word data.
    num_words (int): Number of words to recommend.
    basic_mastery (float): Basic mastery level.
    is_new_user (bool): Whether the user is new.

    Returns:
    pd.DataFrame: DataFrame containing selected words.
    """
    num_basic_words = max(5, int(num_words * 0.5)) if is_new_user else max(3, int(num_words * 0.3))
    num_other_words = num_words - num_basic_words

    # Filtrer les mots de niveau 3 pour les nouveaux utilisateurs
    if is_new_user:
        df = df[df['niv_diff_id'] <= 2]

    # Éliminer les doublons basés sur le mot lui-même
    df = df.drop_duplicates(subset=['mot'])

    # Ajout d'un facteur aléatoire plus important
    df['random_factor'] = np.random.rand(len(df))
    df['recommendation_score'] = df['recommendation_score'] * 0.7 + df['random_factor'] * 0.3

    basic_words = df[df['category'] == 'basique'].nlargest(num_basic_words, 'recommendation_score')
    other_words = df[df['category'] != 'basique'].nlargest(num_other_words, 'recommendation_score')

    if len(basic_words) < num_basic_words:
        logging.warning(f"Seulement {len(basic_words)} mots basiques trouvés sur {num_basic_words} demandés.")
        other_words = other_words.nlargest(num_words - len(basic_words), 'recommendation_score')

    recommended_words = pd.concat([basic_words, other_words])

    # Si après l'élimination des doublons nous avons moins de mots que demandé,
    # compléter avec d'autres mots
    if len(recommended_words) < num_words:
        remaining_words = df[~df['mot_id'].isin(recommended_words['mot_id'])]
        additional_words = remaining_words.nlargest(num_words - len(recommended_words), 'recommendation_score')
        recommended_words = pd.concat([recommended_words, additional_words])

    return recommended_words.nlargest(num_words, 'recommendation_score').sample(frac=1)  # Mélanger les résultats

def analyze_recommendations(recommendations):
    """
    Analyzes the recommendations and prints various statistics.

    Parameters:
    recommendations (pd.DataFrame): DataFrame containing recommended words.
    """
    print("\nAnalyse des recommandations:")
    print(f"Niveau de difficulté moyen: {recommendations['niv_diff_id'].mean():.2f}")
    print(f"Score de prédiction moyen: {recommendations['prediction'].mean():.2f}")
    print(f"Score de difficulté moyen: {recommendations['difficulty_score'].mean():.2f}")
    print(f"Score de récence moyen: {recommendations['recency_score'].mean():.2f}")
    print(f"Score de nouveauté moyen: {recommendations['novelty_score'].mean():.2f}")
    print("\nDistribution des catégories:")
    print(recommendations['category'].value_counts())
    print("\nTop 3 mots recommandés:")
    for _, word in recommendations.head(3).iterrows():
        print(f"Mot: {word['mot']}, Catégorie: {word['category']}, Niveau: {word['niv_diff_id']}, Score: {word['recommendation_score']:.4f}")


def get_user_level(conn, user_id):
    """
    Retrieves the user level and related statistics from the database.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.
    user_id (int): User ID.

    Returns:
    tuple: User level, basic mastery, total words seen, max echelon, and whether the user is new.
    """
    query = """
    WITH user_stats AS (
        SELECT 
            COALESCE(AVG(CASE WHEN sq.score THEN 1 ELSE 0 END), 0) as avg_score,
            COUNT(DISTINCT CASE WHEN c.name = 'basique' THEN m.mot_id END) as basic_words_seen,
            SUM(CASE WHEN c.name = 'basique' AND sq.score THEN 1 ELSE 0 END) as basic_words_correct,
            COUNT(DISTINCT m.mot_id) as total_words_seen,
            MAX(e.echelon_id) as max_echelon
        FROM score_quiz sq
        JOIN quiz q ON sq.quiz_id = q.quiz_id
        JOIN mot m ON sq.mot_id = m.mot_id
        JOIN mot_categorie mc ON m.mot_id = mc.mot_id
        JOIN categorie c ON mc.categorie_id = c.categorie_id
        LEFT JOIN echelon_db e ON m.echelon_id = e.echelon_id
        WHERE q.user_id = %s
    ), total_basic AS (
        SELECT COUNT(*) as total_basic_words
        FROM mot m
        JOIN mot_categorie mc ON m.mot_id = mc.mot_id
        JOIN categorie c ON mc.categorie_id = c.categorie_id
        WHERE c.name = 'basique'
    )
    SELECT *, (SELECT total_basic_words FROM total_basic) as total_basic_words
    FROM user_stats
    """
    df = pd.read_sql(query, conn, params=(user_id,))

    avg_score = df['avg_score'].iloc[0] if not df.empty else 0
    basic_words_seen = df['basic_words_seen'].iloc[0] if not df.empty else 0
    basic_words_correct = df['basic_words_correct'].iloc[0] if not df.empty else 0
    total_words_seen = df['total_words_seen'].iloc[0] if not df.empty else 0
    max_echelon = df['max_echelon'].iloc[0] if not df.empty else None
    total_basic_words = df['total_basic_words'].iloc[0] if not df.empty and df['total_basic_words'].iloc[0] > 0 else 1

    # Calcul pondéré de la maîtrise de base
    if total_basic_words > 0 and basic_words_seen > 0:
        basic_mastery = (basic_words_correct / total_basic_words) * (basic_words_seen / total_basic_words)
    else:
        basic_mastery = 0

    is_new_user = total_words_seen < 10

    if is_new_user:
        level = 1
    else:
        level = min(5, max(1, int(avg_score * 5) + 1))

    return level, basic_mastery, total_words_seen, max_echelon, is_new_user


def update_user_history(conn, user_id, mot_id, correct):
    """
    Updates the user word history in the database.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.
    user_id (int): User ID.
    mot_id (int): Word ID.
    correct (bool): Whether the user answered correctly.
    """
    query = """
    INSERT INTO user_word_history (user_id, mot_id, times_seen, times_correct, last_seen)
    VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP)
    ON CONFLICT (user_id, mot_id) DO UPDATE
    SET times_seen = user_word_history.times_seen + 1,
        times_correct = user_word_history.times_correct + %s,
        last_seen = CURRENT_TIMESTAMP;
    """
    with conn.begin() as transaction:
        conn.execute(query, (user_id, mot_id, int(correct), int(correct)))


def generate_daily_quiz(conn, user_id, pipeline, num_words=10):
    """
    Generates a daily quiz for the user based on the trained model.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.
    user_id (int): User ID.
    pipeline (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    num_words (int): Number of words to include in the quiz.

    Returns:
    pd.DataFrame: DataFrame containing quiz words.
    """
    recommendations = get_word_recommendations(conn, user_id, pipeline, num_words)
    return recommendations


def evaluate_quiz_performance(conn, user_id, quiz_results):
    """
    Evaluates the user's performance on the quiz.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.
    user_id (int): User ID.
    quiz_results (dict): Dictionary containing quiz results.

    Returns:
    float: Performance score.
    """
    correct_count = sum(quiz_results.values())
    total_count = len(quiz_results)
    performance = correct_count / total_count

    for mot_id, correct in quiz_results.items():
        update_user_history(conn, user_id, mot_id, correct)

    return performance


def plot_learning_curve(conn, user_id):
    """
    Plots the learning curve for the user.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.
    user_id (int): User ID.
    """
    query = """
    SELECT DATE(q.date) as quiz_date, AVG(sq.score::int) as avg_score
    FROM quiz q
    JOIN score_quiz sq ON q.quiz_id = sq.quiz_id
    WHERE q.user_id = %s
    GROUP BY DATE(q.date)
    ORDER BY DATE(q.date)
    """
    df = pd.read_sql(query, conn, params=[user_id])

    plt.figure(figsize=(10, 6))
    plt.plot(df['quiz_date'], df['avg_score'], marker='o')
    plt.title(f"Learning curve for the user {user_id}")
    plt.xlabel("Date")
    plt.ylabel("Average score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'learning_curve_user_{user_id}.png')
    plt.close()


def get_feature_names(preprocessor):
    """
    Retrieves the feature names from the preprocessor.

    Parameters:
    preprocessor (sklearn.compose.ColumnTransformer): Preprocessor object used for data transformation.

    Returns:
    list: List of feature names.
    """
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            feature_names.extend(transformer.get_feature_names_out(features))
        elif name == 'cyc':
            feature_names.extend(features)
    return feature_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_recommendation_distribution(recommendations):
    """
    Plots the distribution of categories in the recommendations.

    Parameters:
    recommendations (pd.DataFrame): DataFrame containing recommended words.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(x='category', data=recommendations)
    plt.title('Distribution of categories in the recommendations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('recommendation_distribution.png')
    plt.close()

def plot_difficulty_prediction_distribution(recommendations):
    """
    Plots the distribution of difficulty levels and predictions in the recommendations.

    Parameters:
    recommendations (pd.DataFrame): DataFrame containing recommended words.
    """
    plt.figure(figsize=(14, 6))

    # Sub-plot for the distribution of difficulty levels
    plt.subplot(1, 2, 1)
    sns.histplot(recommendations['niv_diff_id'], bins=range(1, 7), kde=False, color='blue')
    plt.title('Distribution of difficulty levels')
    plt.xlabel('Level of difficulty')
    plt.ylabel('Frequency')

    # Sous-plot pour la distribution des prédictions
    plt.subplot(1, 2, 2)
    sns.histplot(recommendations['prediction'], kde=True, color='green')
    plt.title('Distribution of probability predictions')
    plt.xlabel('Prediction')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig('difficulty_prediction_distribution.png')
    plt.close()

def prepare_recommendations_json(recommendations):
    """
    Prepares the recommendations as a JSON string.

    Parameters:
    recommendations (pd.DataFrame): DataFrame containing recommended words.

    Returns:
    str: JSON string of recommendations.
    """
    return json.dumps(recommendations.to_dict(orient='records'))


def main():
    """
    Main function to execute the data preparation, model training, and word recommendation process.
    """
    logging.info("Starting the HybridReco_final script")
    try:
        conn = get_db_connection()
        logging.info("Database connection established")

        # Data preparation
        df = prepare_training_data(conn)
        X, y, preprocessor = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training and assessment of the model
        pipeline, study = train_model(X_train, y_train, preprocessor)
        metrics = evaluate_model(pipeline, X_test, y_test)
        logging.info("Completed model training")
        logging.info(f"Model metrics: {metrics}")

        # Interpreting the model
        feature_names = get_feature_names(preprocessor)
        plot_feature_importance(pipeline, feature_names)

        # Save results
        save_results(pipeline, metrics, study)

        try:
            # Test for existing users
            existing_user_id = "ff96ac45-db30-4055-b12e-1b52555c66e8"
            logging.info(f"Recommendations for existing user {existing_user_id}:")
            recommendations = get_word_recommendations(conn, existing_user_id, pipeline, num_words=10)
            print("Recommendations for the test user:")
            print(recommendations)
            analyze_recommendations(recommendations)

            # View the distribution of categories
            plot_recommendation_distribution(recommendations)

            # View the distribution of difficulty levels and predictions
            plot_difficulty_prediction_distribution(recommendations)

            # plot user prog
            plot_learning_curve(conn, user_id=existing_user_id)

            #  Prepare for JSON export
            json_recommendations = prepare_recommendations_json(recommendations)
            print("\nJSON recommendations:")
            print(json_recommendations)

            # Test avec un nouvel utilisateur
            new_user_id = "f476a395-ccaf-4122-9365-5b771174bbc6"
            logging.info(f"Recommendations for users with no history {new_user_id}:")
            new_user_recommendations = get_word_recommendations(conn, new_user_id, pipeline)
            print(new_user_recommendations)
            analyze_recommendations(new_user_recommendations)

            # Visualiser la distribution pour le nouvel utilisateur
            plot_difficulty_prediction_distribution(new_user_recommendations)

        except Exception as e:

            logging.error(f"Script successfully completed {str(e)}", exc_info=True)

        finally:
            logging.info("Script terminé avec succès")
    except Exception as e:
        logging.error(f"An error has occurred:  {str(e)}", exc_info=True)




if __name__ == "__main__":
    main()