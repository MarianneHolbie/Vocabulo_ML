
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Vocabulo_quiz.src.db_connexion.database import get_db_connection


def load_data(engine):
    """
    Load data from the database using the provided SQLAlchemy engine.

    Parameters:
    engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.

    Returns:
    tuple: DataFrames containing data from the authentication, mot, quiz, score_quiz, and eval_mot tables.
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
    Prepare user features by calculating various statistics and merging them into a single DataFrame.

    Parameters:
    users_df (pd.DataFrame): DataFrame containing user data.
    quiz_df (pd.DataFrame): DataFrame containing quiz data.
    score_df (pd.DataFrame): DataFrame containing score data.
    eval_df (pd.DataFrame): DataFrame containing evaluation data.

    Returns:
    pd.DataFrame: DataFrame containing user features.
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
    Prepare word features by calculating various statistics and merging them into a single DataFrame.

    Parameters:
    words_df (pd.DataFrame): DataFrame containing word data.
    score_df (pd.DataFrame): DataFrame containing score data.

    Returns:
    pd.DataFrame: DataFrame containing word features.
    """
    # Calculate the average difficulty of each word
    word_difficulty = score_df.groupby('mot_id')['score'].mean().reset_index(name='avg_difficulty')
    word_difficulty['avg_difficulty'] = 1 - word_difficulty[
        'avg_difficulty']  # Invert to indicate higher difficulty

    # Calculate the frequency of each word
    word_frequency = score_df['mot_id'].value_counts().reset_index(name='frequency')
    word_frequency.columns = ['mot_id', 'frequency']

    # Merge with basic word information
    word_features = words_df.merge(word_difficulty, on='mot_id', how='left')
    word_features = word_features.merge(word_frequency, on='mot_id', how='left')

    # Encode grammatical categories (if necessary)
    word_features = pd.get_dummies(word_features, columns=['gramm_id'], prefix='gram')

    # Fill missing values
    word_features = word_features.fillna(0)

    return word_features


def create_user_word_pairs(quiz_df, score_df):
    """
    Create user-word pairs by merging quiz and score data.

    Parameters:
    quiz_df (pd.DataFrame): DataFrame containing quiz data.
    score_df (pd.DataFrame): DataFrame containing score data.

    Returns:
    pd.DataFrame: DataFrame containing user-word pairs with scores.
    """
    # Merge quiz and score DataFrames
    pairs = pd.merge(quiz_df, score_df, on='quiz_id')

    print("Available columns in pairs:", pairs.columns)

    # Resolve the issue of duplicate mot_id
    if 'mot_id_x' in pairs.columns and 'mot_id_y' in pairs.columns:
        # Use mot_id_y from score_quiz
        pairs['mot_id'] = pairs['mot_id_y']
        # Drop old columns
        pairs = pairs.drop(['mot_id_x', 'mot_id_y'], axis=1)
    elif 'mot_id' not in pairs.columns:
        raise ValueError("The 'mot_id' column is not present in the merged DataFrame.")

    # Check if all necessary columns are present
    required_columns = ['user_id', 'mot_id', 'score']
    missing_columns = [col for col in required_columns if col not in pairs.columns]

    if missing_columns:
        raise ValueError(f"Missing columns after correction: {missing_columns}")

    # Select necessary columns
    pairs = pairs[required_columns]

    return pairs


def normalize_features(df, columns_to_normalize):
    """
    Normalize specified columns in the DataFrame using StandardScaler.

    Parameters:
    df (pd.DataFrame): DataFrame containing features to be normalized.
    columns_to_normalize (list): List of column names to be normalized.

    Returns:
    tuple: DataFrame with normalized features and the fitted scaler.
    """
    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler

def prepare_data_for_model():
    """
    Prepare data for model training by loading, processing, and merging various features.

    Returns:
    tuple: Features (X), labels (y), user features, and word features.
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

    # Separate features and labels
    X = final_data.drop('score', axis=1)
    y = final_data['score']

    return X, y, user_features, word_features

if __name__ == "__main__":
    X, y, user_features, word_features = prepare_data_for_model()
    print("Data preparation complete.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"user_features shape: {user_features.shape}")
    print(f"word_features shape: {word_features.shape}")