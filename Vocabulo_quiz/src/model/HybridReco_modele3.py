import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import optuna
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from Vocabulo_quiz.src.db_connexion.database import get_db_connection
import logging

OUTPUT_DIR = '/app/output2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration loggin
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'hybrid_reco_v2.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_training_data(conn):
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
    # Traitement des colonnes temporelles
    df['quiz_date'] = pd.to_datetime(df['quiz_date'])
    df['hour'] = df['quiz_date'].dt.hour
    df['day_of_week'] = df['quiz_date'].dt.dayofweek
    df['month'] = df['quiz_date'].dt.month

    # Encodage cyclique du temps
    for col in ['hour', 'day_of_week', 'month']:
        max_val = 24 if col == 'hour' else 7 if col == 'day_of_week' else 12
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    # Définition des types de features
    numeric_features = ['frequence', 'times_correct', 'times_seen', 'days_since_last_seen',
                        'freqfilms', 'freqlivres', 'nbr_syll', 'cp_cm2_sfi']
    categorical_features = ['niv_diff_id', 'gramm_id', 'category', 'subcategory', 'echelon_id', 'user_feedback']
    cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']

    # Création du préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('cyc', 'passthrough', cyclical_features)
        ])

    # Préparation des features et de la target
    X = df.drop(['score', 'user_id', 'mot_id', 'quiz_date'], axis=1)
    y = df['score']

    return X, y, preprocessor


def objective(trial, X, y, preprocessor):
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
    }

    model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    return scores.mean()


def train_model(X, y, preprocessor):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=100)

    best_params = study.best_params
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X, y)

    return pipeline, study


def evaluate_model(pipeline, X_test, y_test):
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

    return metrics


def plot_feature_importance(pipeline, feature_names):
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
    output_dir = os.path.join(os.path.dirname(__file__), 'output2')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()


def interpret_model(pipeline, X, feature_names):
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(pipeline.named_steps['preprocessor'].transform(X))

    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, pipeline.named_steps['preprocessor'].transform(X), feature_names=feature_names,
                      plot_type="bar")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()


def save_model(pipeline, metrics, study):
    joblib.dump(pipeline, 'xgboost_pipeline.joblib')

    with open('model_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

    joblib.dump(study, 'optuna_study.joblib')


def save_results(pipeline, metrics, study):
    output_dir = os.path.join(os.path.dirname(__file__), 'output2')
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, 'xgboost_pipeline_v2.joblib'))

    try:
        joblib.dump(pipeline, os.path.join(output_dir, 'xgboost_pipeline_v2.joblib'))
        logging.info(f"Pipeline sauvegardé dans {output_dir}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du pipeline : {str(e)}")

    try:
        with open(os.path.join(output_dir, 'model_metrics_v2.txt'), 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        logging.info(f"Métriques sauvegardées dans {output_dir}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des métriques : {str(e)}")

    try:
        joblib.dump(study, os.path.join(output_dir, 'optuna_study_v2.joblib'))
        logging.info(f"Étude Optuna sauvegardée dans {output_dir}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde de l'étude Optuna : {str(e)}")

    logging.info(f"Tous les résultats ont été sauvegardés dans {output_dir}")


def load_model():
    pipeline = joblib.load('xgboost_pipeline.joblib')
    return pipeline


def calculate_forgetting_curve(strength, time):
    return np.exp(-time / strength)


def calculate_optimal_review_time(strength, threshold=0.5):
    return -strength * np.log(threshold)


def get_word_recommendations(conn, user_id, pipeline, num_words=10):
    user_level, basic_mastery, total_words_seen, _ = get_user_level(conn, user_id)
    logging.info(f"User level: {user_level}, Basic mastery: {basic_mastery}, Total words seen: {total_words_seen}")

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

    # Prétraitement des données
    df['hour'] = pd.to_datetime(df['current_time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['current_time']).dt.dayofweek
    df['month'] = pd.to_datetime(df['current_time']).dt.month

    for col in ['hour', 'day_of_week', 'month']:
        max_val = 24 if col == 'hour' else 7 if col == 'day_of_week' else 12
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)

    # Ajouter une colonne de feedback fictif (puisque c'est une prédiction)
    df['user_feedback'] = 'Bien'

    X_pred = df.drop(['mot_id', 'mot', 'current_time'], axis=1)

    # Prédire la probabilité de réponse correcte
    probabilities = pipeline.predict_proba(X_pred)[:, 1]

    df['prediction'] = probabilities

    # Calcul du score de recommandation ajusté
    df['recommendation_score'] = df.apply(
        lambda row: calculate_recommendation_score(row, user_level, basic_mastery, total_words_seen),
        axis=1
    )

    # Sélection des mots recommandés
    recommended_words = select_diverse_words(df, num_words, basic_mastery)
    logging.info(f"Selected {len(recommended_words)} words for recommendation")

    return recommended_words[['mot_id', 'mot', 'category', 'subcategory', 'niv_diff_id', 'recommendation_score']]


def calculate_recommendation_score(row, user_level, basic_mastery, total_words_seen):
    prediction = row['prediction']
    difficulty = row['niv_diff_id']
    times_seen = row['times_seen']
    days_since_last_seen = row['days_since_last_seen']
    is_basic = row['category'] == 'basique'
    frequence = row['frequence']

    logging.debug(f"Calculating score for word: {row['mot']}")
    logging.debug(
        f"Input values: difficulty={difficulty}, times_seen={times_seen}, days_since_last_seen={days_since_last_seen}, frequence={frequence}")

    # Favoriser les mots avec une difficulté appropriée
    difficulty_factor = 1 - abs(difficulty - user_level) / max(user_level, 5)

    # Favoriser les mots moins vus
    novelty = 1 / (times_seen + 1)

    # Facteur de récence (éviter la division par zéro)
    recency = 1 / (days_since_last_seen + 1)

    # Utiliser le score de difficulté (frequence)
    difficulty_score = 1 - (frequence / 100) if frequence is not None and frequence != 0 else 0.5

    # Bonus pour les mots basiques si la maîtrise de base est faible
    basic_bonus = 1.5 if is_basic and basic_mastery < 0.7 else 1

    # Facteur basé sur la fréquence dans les films et les livres
    frequency_factor = (row['freqfilms'] + row['freqlivres']) / 2
    frequency_score = 1 - (frequency_factor / 100) if frequency_factor != 0 else 0.5

    # Facteur basé sur le nombre de syllabes
    syllable_factor = 1 / (row['nbr_syll'] + 1)

    # Calculer le score final
    score = (
        difficulty_factor * 0.2 +
        novelty * 0.15 +
        recency * 0.1 +
        difficulty_score * 0.15 +
        frequency_score * 0.15 +
        syllable_factor * 0.05 +
        (1 - prediction) * 0.2  # Favoriser les mots que le modèle prédit comme plus difficiles
    ) * basic_bonus

    logging.debug(f"Calculated score: {score}")
    return score


def select_diverse_words(df, num_words, basic_mastery):
    # Déterminer la proportion de mots basiques à inclure
    basic_proportion = max(0.3, 1 - basic_mastery)  # Au moins 30% de mots basiques, plus si la maîtrise est faible
    num_basic_words = max(1, int(num_words * basic_proportion))
    num_other_words = num_words - num_basic_words

    logging.info(f"Selecting {num_basic_words} basic words and {num_other_words} other words")

    # Sélectionner les mots basiques
    basic_words = df[df['category'] == 'basique'].nlargest(num_basic_words, 'recommendation_score')

    # Sélectionner les autres mots
    other_words = df[df['category'] != 'basique'].nlargest(num_other_words, 'recommendation_score')

    # Combiner et mélanger les résultats
    recommended_words = pd.concat([basic_words, other_words]).sample(frac=1)

    return recommended_words


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
    df = pd.read_sql(query, conn, params=(user_id,))

    if df.empty:
        print(f"Aucune donnée trouvée pour l'utilisateur {user_id}")
        return 1, 0, 0, 0  # Valeurs par défaut si aucune donnée n'est trouvée

    avg_score = df['avg_score'].iloc[0]
    basic_words_seen = df['basic_words_seen'].iloc[0]
    basic_words_correct = df['basic_words_correct'].iloc[0]
    total_words_seen = df['total_words_seen'].iloc[0]
    max_echelon = df['max_echelon'].iloc[0]

    basic_mastery = basic_words_correct / basic_words_seen if basic_words_seen > 0 else 0
    level = min(5, max(1, int(avg_score * 5) + 1))  # Niveau de 1 à 5

    return level, basic_mastery, total_words_seen, max_echelon


def update_user_history(conn, user_id, mot_id, correct):
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
    recommendations = get_word_recommendations(conn, user_id, pipeline, num_words)
    return recommendations


def evaluate_quiz_performance(conn, user_id, quiz_results):
    correct_count = sum(quiz_results.values())
    total_count = len(quiz_results)
    performance = correct_count / total_count

    for mot_id, correct in quiz_results.items():
        update_user_history(conn, user_id, mot_id, correct)

    return performance


def plot_learning_curve(conn, user_id):
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
    plt.title(f"Courbe d'apprentissage pour l'utilisateur {user_id}")
    plt.xlabel("Date")
    plt.ylabel("Score moyen")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'learning_curve_user_{user_id}.png')
    plt.close()


def get_feature_names(preprocessor):
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
    plt.figure(figsize=(12, 6))
    sns.countplot(x='category', data=recommendations)
    plt.title('Distribution des catégories dans les recommandations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('recommendation_distribution.png')
    plt.close()


def main():
    logging.info("Démarrage du script HybridReco_v2")
    try:
        conn = get_db_connection()
        logging.info("Connexion à la base de données établie")

        # Préparation des données
        df = prepare_training_data(conn)
        X, y, preprocessor = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entraînement et évaluation du modèle
        pipeline, study = train_model(X_train, y_train, preprocessor)
        metrics = evaluate_model(pipeline, X_test, y_test)
        logging.info("Entraînement du modèle terminé")
        logging.info(f"Métriques du modèle : {metrics}")

        # Interprétation du modèle
        feature_names = get_feature_names(preprocessor)
        plot_feature_importance(pipeline, feature_names)

        # Sauvegarde des résultats
        save_results(pipeline, metrics, study)

        # Génération de recommandations pour un utilisateur test
        user_id = "ff96ac45-db30-4055-b12e-1b52555c66e8"  # existing user
        try:
            recommendations = get_word_recommendations(conn, user_id, pipeline, num_words=10)
            logging.info("Recommandations générées:")
            logging.info(recommendations.to_string())

            plot_recommendation_distribution(recommendations)

            # Analyse des recommandations
            basic_words = recommendations[recommendations['category'] == 'basique']
            logging.info(f"Proportion de mots basiques: {len(basic_words) / len(recommendations):.2f}")
            logging.info(
                f"Niveaux de difficulté recommandés: {recommendations['niv_diff_id'].value_counts().to_dict()}")

            print("Recommandations pour l'utilisateur test:")
            print(recommendations)
        except Exception as e:
            print(f"Erreur lors de la génération des recommandations : {str(e)}")

        logging.info("Script terminé avec succès")
        conn.close()
    except Exception as e:
        logging.error(f"Une erreur s'est produite : {str(e)}", exc_info=True)




if __name__ == "__main__":
    main()