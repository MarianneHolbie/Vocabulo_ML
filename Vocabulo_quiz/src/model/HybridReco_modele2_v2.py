import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ndcg_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import optuna
from gensim.models import Word2Vec
import warnings
from Vocabulo_quiz.src.db_connexion.database import get_db_connection
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def prepare_data(conn):
    """
    Prepares the data by executing a SQL query and returning the result as a DataFrame.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): Database connection object.

    Returns:
    pd.DataFrame: DataFrame containing the prepared data.
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
    """
     Preprocesses the data by handling temporal columns, extracting features, and applying transformations.

     Parameters:
     df (pd.DataFrame): DataFrame containing the raw data.

     Returns:
     tuple: Transformed features (X), target variable (y), feature names, and preprocessor object.
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

    # New feature: discretized days_since_last_seen
    df['days_since_last_seen_cat'] = pd.cut(df['days_since_last_seen'],
                                            bins=[0, 7, 30, np.inf],
                                            labels=['recent', 'medium', 'old'])

    # Word Embedding
    sentences = df['mot'].apply(lambda x: x.split()).tolist()
    word2vec_model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    word_embeddings = df['mot'].apply(lambda x: word2vec_model.wv[x] if x in word2vec_model.wv else np.zeros(100))

    # Adding embeddings to the DataFrame
    word_embedding_cols = [f'embed_{i}' for i in range(100)]
    word_embedding_df = pd.DataFrame(word_embeddings.tolist(), columns=word_embedding_cols)
    df = pd.concat([df, word_embedding_df], axis=1)

    # Application of PCA on embeddings
    pca = PCA(n_components=10)
    pca_embeddings = pca.fit_transform(df[word_embedding_cols])
    pca_cols = [f'embed_pca_{i}' for i in range(10)]
    df[pca_cols] = pca_embeddings

    # Identify column types
    numeric_features = ['frequence', 'times_correct', 'times_seen', 'freqfilms', 'freqlivres', 'nbr_syll', 'cp_cm2_sfi',
                        'days_since_last_seen']
    categorical_features = ['niv_diff_id', 'gramm_id', 'category', 'subcategory', 'echelon_id',
                            'days_since_last_seen_cat']
    cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
    embedding_features = [f'embed_pca_{i}' for i in range(10)]

    # Process UUID columns
    uuid_columns = ['user_id', 'mot_id']
    for col in uuid_columns:
        df[col] = df[col].astype(str)

    #  Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features + uuid_columns),
            ('cyc', 'passthrough', cyclical_features),
            ('emb', 'passthrough', embedding_features)
        ])

    # Apply pre-treatment
    X = preprocessor.fit_transform(df)
    y = df['score'].astype(int)

    ## Obtain feature names after pre-processing
    feature_names = (numeric_features +
                     preprocessor.named_transformers_['cat'].get_feature_names_out(
                         categorical_features + uuid_columns).tolist() +
                     cyclical_features +
                     embedding_features)

    return X, y, feature_names, preprocessor


def create_model(trial, model_type, input_dim):
    """
    Creates a machine learning model based on the specified type and trial parameters.

    Parameters:
    trial (optuna.trial.Trial): Optuna trial object for hyperparameter optimization.
    model_type (str): Type of model to create ('rf', 'xgb', 'nn').
    input_dim (int): Number of input features.

    Returns:
    model: Created machine learning model.
    """
    if model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            class_weight='balanced'
        )
    elif model_type == 'xgb':
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            scale_pos_weight=trial.suggest_float('scale_pos_weight', 1e-1, 1e1, log=True)
        )
    elif model_type == 'nn':
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(trial.suggest_int('units_1', 32, 256), activation='relu'),
            Dropout(trial.suggest_float('dropout_1', 0.1, 0.5)),
            Dense(trial.suggest_int('units_2', 16, 128), activation='relu'),
            Dropout(trial.suggest_float('dropout_2', 0.1, 0.5)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def objective(trial, X, y, model_type):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    trial (optuna.trial.Trial): Optuna trial object.
    X (np.array): Feature matrix.
    y (np.array): Target variable.
    model_type (str): Type of model to create ('rf', 'xgb', 'nn').

    Returns:
    float: F1 score of the model.
    """
    model = create_model(trial, model_type, X.shape[1])
    if model_type in ['rf', 'xgb']:
        model.fit(X, y)
        y_pred = model.predict(X)
    else:  # neural network
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        y_pred = (model.predict(X) > 0.5).astype(int)
    return f1_score(y, y_pred)


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='rf'):
    """
    Trains and evaluates a machine learning model using Optuna for hyperparameter optimization.

    Parameters:
    X_train (np.array): Training feature matrix.
    y_train (np.array): Training target variable.
    X_test (np.array): Testing feature matrix.
    y_test (np.array): Testing target variable.
    model_type (str): Type of model to create ('rf', 'xgb', 'nn').

    Returns:
    tuple: Best model and its performance metrics.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, model_type), n_trials=100)

    best_model = create_model(study.best_trial, model_type, X_train.shape[1])
    if model_type in ['rf', 'xgb']:
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    else:  # neural network
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        y_pred = (best_model.predict(X_test) > 0.5).astype(int)
        y_pred_proba = best_model.predict(X_test).flatten()

    # Model evaluation
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred_proba),
        "NDCG": ndcg_score([y_test], [y_pred_proba])
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return best_model, metrics


def visualize_results(metrics, model_type):
    """
    Visualizes the performance metrics of a model using a bar plot.

    Parameters:
    metrics (dict): Dictionary containing performance metrics.
    model_type (str): Type of model.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(f"Performance Metrics for {model_type.upper()} Model")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{model_type}_performance_metrics.png")
    plt.close()


def save_model(model, preprocessor, model_type):
    """
    Saves the trained model and preprocessor to disk.

    Parameters:
    model: Trained machine learning model.
    preprocessor: Preprocessor object used for data transformation.
    model_type (str): Type of model.
    """

    joblib.dump(model, f"{model_type}_model.joblib")

    joblib.dump(preprocessor, f"{model_type}_preprocessor.joblib")
    print(f"Model and preprocessor saved as {model_type}_model.joblib and {model_type}_preprocessor.joblib")

if __name__ == "__main__":
    conn = get_db_connection()
    df = prepare_data(conn)
    X, y, feature_names, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_models = {}
    best_metrics = {}

    for model_type in ['rf', 'xgb', 'nn']:
        print(f"\nTraining {model_type.upper()} model:")
        best_model, metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type)
        best_models[model_type] = best_model
        best_metrics[model_type] = metrics

        visualize_results(metrics, model_type)
        save_model(best_model, preprocessor, model_type)

    #  Compare model performance
    best_model_type = max(best_metrics, key=lambda k: best_metrics[k]['F1-score'])
    print(f"\nBest performing model: {best_model_type.upper()}")
    print("Best model metrics:")
    for metric, value in best_metrics[best_model_type].items():
        print(f"{metric}: {value:.4f}")

    # Save the best model and its preprocessor
    save_model(best_models[best_model_type], preprocessor, "best_model")

    conn.close()