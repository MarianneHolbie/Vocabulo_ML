import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from Vocabulo_quiz.src.db_connexion.database import get_db_connection


def load_data(engine):
    """
    Loads data from the database using the provided engine.

    :param engine: sqlalchemy.engine.base.Engine - The database engine to use for the connection.
    :return: tuple - A tuple containing DataFrames for users, words, quizzes, scores, and evaluations.
    """
    users_df = pd.read_sql("SELECT * FROM authentication", engine)
    words_df = pd.read_sql("SELECT * FROM mot", engine)
    quiz_df = pd.read_sql("SELECT * FROM quiz", engine)
    score_df = pd.read_sql("SELECT * FROM score_quiz", engine)
    eval_df = pd.read_sql("SELECT * FROM eval_mot", engine)

    print("Data loaded successfully.")
    print("Quiz data preview:")
    print(quiz_df.head())
    print("\nScore data preview:")
    print(score_df.head())

    return users_df, words_df, quiz_df, score_df, eval_df


def prepare_user_features(users_df, quiz_df, score_df, eval_df):
    """
    Prepares user features for the hybrid recommender model.

    This function calculates various features for each user based on their quiz participation, scores, and evaluations.

    :param users_df: pandas.DataFrame - DataFrame containing user information.
    :param quiz_df: pandas.DataFrame - DataFrame containing quiz information.
    :param score_df: pandas.DataFrame - DataFrame containing quiz scores.
    :param eval_df: pandas.DataFrame - DataFrame containing word evaluations.
    :return: pandas.DataFrame - DataFrame containing the prepared user features.
    """
    # Calculate the number of quizzes per user
    quiz_count = quiz_df.groupby('user_id').size().reset_index(name='quiz_count')

    # Calculate the average score per user
    user_scores = score_df.merge(quiz_df, on='quiz_id')
    avg_score = user_scores.groupby('user_id')['score'].mean().reset_index(name='avg_score')

    # Calculate the distribution of evaluations per user
    user_evals = eval_df.merge(quiz_df, on='quiz_id')
    eval_dist = user_evals.groupby(['user_id', 'scale']).size().unstack(fill_value=0)
    eval_dist = eval_dist.div(eval_dist.sum(axis=1), axis=0)

    # Merge all features
    user_features = users_df.merge(quiz_count, on='user_id', how='left')
    user_features = user_features.merge(avg_score, on='user_id', how='left')
    user_features = user_features.merge(eval_dist, on='user_id', how='left')

    # Fill missing values
    user_features = user_features.fillna(0)

    return user_features


def prepare_word_features(words_df, score_df):
    """
    Prepares word features for the hybrid recommender model.

    This function calculates various features for each word based on their difficulty and frequency.

    :param words_df: pandas.DataFrame - DataFrame containing word information.
    :param score_df: pandas.DataFrame - DataFrame containing quiz scores.
    :return: pandas.DataFrame - DataFrame containing the prepared word features.
    """
    # Calculate the average difficulty of each word
    word_difficulty = score_df.groupby('mot_id')['score'].mean().reset_index(name='avg_difficulty')
    word_difficulty['avg_difficulty'] = 1 - word_difficulty[
        'avg_difficulty']  # Inverser pour que les valeurs plus élevées indiquent une plus grande difficulté

    # Calculate the frequency of each word
    word_frequency = score_df['mot_id'].value_counts().reset_index(name='frequency')
    word_frequency.columns = ['mot_id', 'frequency']

    # Merge with basic word information
    word_features = words_df.merge(word_difficulty, on='mot_id', how='left')
    word_features = word_features.merge(word_frequency, on='mot_id', how='left')

    # Encode grammatical categories
    word_features = pd.get_dummies(word_features, columns=['gramm_id'], prefix='gram')

    # Fill missing values
    word_features = word_features.fillna(0)

    return word_features


def create_user_word_pairs(quiz_df, score_df):
    """
    Creates user-word pairs for the hybrid recommender model.

    This function merges quiz and score data to create pairs of users and words with their associated scores.

    :param quiz_df: pandas.DataFrame - DataFrame containing quiz information.
    :param score_df: pandas.DataFrame - DataFrame containing quiz scores.
    :return: pandas.DataFrame - DataFrame containing user-word pairs with scores.
    """
    # Merge quiz and score DataFrames
    pairs = pd.merge(quiz_df, score_df, on='quiz_id')

    print("Colonnes disponibles dans pairs:", pairs.columns)

    # Resolve the issue of duplicate mot_id
    if 'mot_id_x' in pairs.columns and 'mot_id_y' in pairs.columns:
        # Use mot_id_y from score_quiz
        pairs['mot_id'] = pairs['mot_id_y']
        # Drop old columns
        pairs = pairs.drop(['mot_id_x', 'mot_id_y'], axis=1)
    elif 'mot_id' not in pairs.columns:
        raise ValueError("La colonne 'mot_id' n'est pas présente dans le DataFrame fusionné.")

    # Check if all necessary columns are present
    required_columns = ['user_id', 'mot_id', 'score']
    missing_columns = [col for col in required_columns if col not in pairs.columns]

    if missing_columns:
        raise ValueError(f"Colonnes manquantes après correction: {missing_columns}")

    # Select necessary columns
    pairs = pairs[required_columns]

    return pairs


def normalize_features(df, columns_to_normalize):
    """
    Normalizes the specified columns in the DataFrame using StandardScaler.

    :param df: pandas.DataFrame - The DataFrame containing the data to be normalized.
    :param columns_to_normalize: list - List of column names to be normalized.
    :return: tuple - A tuple containing the normalized DataFrame and the fitted StandardScaler.
    """
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler


def prepare_data_for_model():
    """
    Prepares data for the hybrid recommender model.

    This function performs the following steps:
    1. Connects to the database.
    2. Loads data from the database.
    3. Prepares user features.
    4. Prepares word features.
    5. Creates user-word pairs.
    6. Normalizes the features.
    7. Merges all the data into a final DataFrame.

    :return: tuple - A tuple containing the processed DataFrame (X), target values (y),
     and the preprocessor used.
"""
    engine = get_db_connection()
    users_df, words_df, quiz_df, score_df, eval_df = load_data(engine)

    user_features = prepare_user_features(users_df, quiz_df, score_df, eval_df)
    word_features = prepare_word_features(words_df, score_df)
    user_word_pairs = create_user_word_pairs(quiz_df, score_df)

    # Normalize features
    user_numeric_columns = ['quiz_count', 'avg_score', 'Trop dur', 'Bien', 'Trop facile']
    user_features, _ = normalize_features(user_features, user_numeric_columns)

    word_numeric_columns = ['avg_difficulty', 'frequency']
    word_features, _ = normalize_features(word_features, word_numeric_columns)

    # Merge all data
    final_data = user_word_pairs.merge(user_features, on='user_id')
    final_data = final_data.merge(word_features, on='mot_id')

    print("Types of columns in final_data:")
    print(final_data.dtypes)

    print("\nUnique values in categorical columns:")
    for col in ['niv_diff_id', 'echelon_id'] + [col for col in final_data.columns if col.startswith('gram_')]:
        print(f"{col}: {final_data[col].unique()}")

    # Check available columns in final_data
    print("Colonnes disponibles dans final_data:", final_data.columns.tolist())

    # Define columns for each feature type
    numeric_features = ['quiz_count', 'avg_score', 'Bien', 'Trop dur', 'Trop facile', 'frequence', 'avg_difficulty',
                        'frequency', 'alphabet_id']
    categorical_features = ['niv_diff_id', 'echelon_id'] + [col for col in final_data.columns if
                                                            col.startswith('gram_')]

    print("Features numériques utilisées:", numeric_features)
    print("Features catégorielles utilisées:", categorical_features)

    # Create a pipeline for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing
    X = final_data[numeric_features + categorical_features]
    y = final_data['score'].astype(int)  # Convert bool to int

    X_processed = preprocessor.fit_transform(X)

    # Create a DataFrame with the new features
    feature_names = (numeric_features +
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())

    X_processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
                                  columns=feature_names)

    # Add user_id and mot_id to X_processed_df
    X_processed_df['user_id'] = final_data['user_id'].values
    X_processed_df['mot_id'] = final_data['mot_id'].values

    return X_processed_df, y, preprocessor


if __name__ == "__main__":
    X, y, preprocessor = prepare_data_for_model()
    print("Data preparation complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Number of features after preprocessing: {X.shape[1]}")
    print(f"Example rows from X:")
    print(X.head())
    print(f"Distribution of y: {np.bincount(y)}")
