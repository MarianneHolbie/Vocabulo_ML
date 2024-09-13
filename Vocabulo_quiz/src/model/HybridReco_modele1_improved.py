import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

from Vocabulo_quiz.src.db_connexion.database import get_db_connection
from Vocabulo_quiz.src.feature_engineering.data_prep_ameliored import prepare_data_for_model

def create_hybrid_recommender(num_users, num_words, input_dim, embedding_size=16):
    """
    Creates a hybrid recommender model using user and word embeddings along with additional features.

    :param num_users: int - The number of unique users.
    :param num_words: int - The number of unique words.
    :param input_dim: int - The dimension of the additional features input.
    :param embedding_size: int, optional - The size of the embeddings for users and words (default is 16).
    :return: tensorflow.keras.Model - The compiled hybrid recommender model.
"""
    user_input = Input(shape=(1,), name='user_input')
    word_input = Input(shape=(1,), name='word_input')
    features_input = Input(shape=(input_dim,), name='features_input')

    user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
    word_embedding = Embedding(num_words, embedding_size, name='word_embedding')(word_input)

    user_embedding_flat = Flatten()(user_embedding)
    word_embedding_flat = Flatten()(word_embedding)

    concatenated = Concatenate()([user_embedding_flat, word_embedding_flat, features_input])

    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(dropout1)
    dropout2 = Dropout(0.3)(dense2)

    output = Dense(1, activation='sigmoid', name='output')(dropout2)

    model = Model(inputs=[user_input, word_input, features_input], outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_and_evaluate_model(X, y, user_mapping, word_mapping, num_users, num_words, n_splits=5):
    """
    Trains and evaluates the hybrid recommender model using cross-validation.

    :param X: pandas.DataFrame - The input features for the model.
    :param y: pandas.Series - The target labels for the model.
    :param user_mapping: dict - A mapping from user IDs to indices.
    :param word_mapping: dict - A mapping from word IDs to indices.
    :param num_users: int - The number of unique users.
    :param num_words: int - The number of unique words.
    :param n_splits: int, optional - The number of splits for cross-validation (default is 5).
    :return: list of tuples - A list containing the validation loss and accuracy for each fold.
"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_performances = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"Training on fold {fold}")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_user_train = np.array([user_mapping[user] for user in X_train['user_id']])
        X_word_train = np.array([word_mapping[word] for word in X_train['mot_id']])
        X_features_train = np.array(X_train.drop(['user_id', 'mot_id'], axis=1).values)

        X_user_val = np.array([user_mapping[user] for user in X_val['user_id']])
        X_word_val = np.array([word_mapping[word] for word in X_val['mot_id']])
        X_features_val = np.array(X_val.drop(['user_id', 'mot_id'], axis=1).values)

        model = create_hybrid_recommender(num_users, num_words, X_features_train.shape[1])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            [X_user_train, X_word_train, X_features_train],
            y_train,
            epochs=50,
            batch_size=64,
            validation_data=([X_user_val, X_word_val, X_features_val], y_val),
            callbacks=[early_stopping],
            verbose=0
        )

        val_loss, val_accuracy = model.evaluate([X_user_val, X_word_val, X_features_val], y_val, verbose=0)
        fold_performances.append((val_loss, val_accuracy, history))

        print(f"Fold {fold} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return fold_performances


def predict_recommendations(model, X_test, user_mapping, word_mapping, conn, num_recommendations=5):
    """
    Generates recommendations for users based on the trained hybrid recommender model.

    :param model: tensorflow.keras.Model - The trained hybrid recommender model.
    :param X_test: pandas.DataFrame - The test dataset containing user and word features.
    :param user_mapping: dict - A mapping from user IDs to indices.
    :param word_mapping: dict - A mapping from word IDs to indices.
    :param conn: sqlalchemy.engine.base.Connection - The database connection to fetch word details.
    :param num_recommendations: int - The number of recommendations to generate for each user (default is 5).
    :return: dict - A dictionary where keys are user IDs and values are lists of recommended words with details.
    """
    user_ids = X_test['user_id'].unique()
    recommendations = {}

    for user_id in user_ids:
        user_data = X_test[X_test['user_id'] == user_id]
        user_index = user_mapping[user_id]
        word_indices = [word_mapping[word_id] for word_id in user_data['mot_id']]

        user_input = np.array([user_index] * len(word_indices))
        word_input = np.array(word_indices)
        features_input = user_data.drop(['user_id', 'mot_id'], axis=1).values

        predictions = model.predict([user_input, word_input, features_input])

        top_indices = np.argsort(predictions.flatten())[-num_recommendations:]

        reverse_word_mapping = {v: k for k, v in word_mapping.items()}
        recommended_word_ids = [int(reverse_word_mapping[idx]) for idx in top_indices]

        # Fetch the details of recommended words
        word_details = get_word_details(conn, recommended_word_ids)

        recommendations[user_id] = [
            {
                'mot_id': word_id,  # Ensure it's an int
                'mot': word_details[word_id]['mot'],
                'niveau_difficulte': word_details[word_id]['niv_diff_id'],
                'categorie': word_details[word_id]['category'],
                'sous_categorie': word_details[word_id]['subcategory']
            }
            for word_id in recommended_word_ids
        ]

    return recommendations


def evaluate_recommendations(model, X_test, y_test, user_mapping, word_mapping, conn, num_recommendations=5):
    """
    Evaluates the recommendations generated by the hybrid recommender model.

    :param model: tensorflow.keras.Model - The trained hybrid recommender model.
    :param X_test: pandas.DataFrame - The test dataset containing user and word features.
    :param y_test: pandas.Series - The true labels for the test dataset.
    :param user_mapping: dict - A mapping from user IDs to indices.
    :param word_mapping: dict - A mapping from word IDs to indices.
    :param conn: sqlalchemy.engine.base.Connection - The database connection to fetch word details.
    :param num_recommendations: int - The number of recommendations to generate for each user (default is 5).
    :return: dict - A dictionary containing evaluation metrics: precision, recall, f1_score, and ndcg.
    """
    recommendations = predict_recommendations(model, X_test, user_mapping, word_mapping, conn, num_recommendations)

    precisions = []
    recalls = []
    ndcgs = []
    f1_scores = []

    for user_id, recommended_words in recommendations.items():
        user_data = X_test[X_test['user_id'] == user_id]
        actual_positive = set(user_data.loc[y_test.loc[user_data.index] == 1, 'mot_id'].astype(int))

        if len(actual_positive) == 0:
            continue  # Skip users with no positive interactions

        recommended_set = set(word['mot_id'] for word in recommended_words)

        # Calculating precision and recall
        precision = len(recommended_set.intersection(actual_positive)) / len(recommended_set)
        recall = len(recommended_set.intersection(actual_positive)) / len(actual_positive)

        precisions.append(precision)
        recalls.append(recall)

        # F1-score calculation
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_scores.append(f1)

        # Preparation for NDCG
        relevance = [1 if word['mot_id'] in actual_positive else 0 for word in recommended_words]
        ideal = sorted(relevance, reverse=True)

        ndcg = ndcg_score([ideal], [relevance])
        ndcgs.append(ndcg)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_ndcg = np.mean(ndcgs)

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'ndcg': avg_ndcg
    }


def get_word_details(conn, recommended_word_ids):
    """
    Fetches details of recommended words from the database.

    Parameters:
    conn (sqlalchemy.engine.base.Connection): The database connection to execute the query.
    recommended_word_ids (list): List of recommended word IDs.

    Returns:
    dict: A dictionary where keys are word IDs and values are dictionaries containing word details.
    """
    placeholders = ','.join(['%s'] * len(recommended_word_ids))
    query = f"""
            SELECT 
                m.mot_id, 
                m.mot, 
                m.niv_diff_id, 
                c.name as category, 
                sc.name as subcategory
            FROM mot m
            LEFT JOIN mot_categorie mc ON m.mot_id = mc.mot_id
            LEFT JOIN categorie c ON mc.categorie_id = c.categorie_id
            LEFT JOIN mot_subcategory ms ON m.mot_id = ms.mot_id
            LEFT JOIN subcategory sc ON ms.subcat_id = sc.subcat_id
            WHERE m.mot_id IN ({placeholders})
            """
    df = pd.read_sql(query, conn, params=tuple(recommended_word_ids))  # Convert list to tuple
    return df.set_index('mot_id').T.to_dict()


def plot_history(histories):
    """
    Plot the training and validation loss and accuracy from the histories of multiple folds.

    :param histories: list of History objects - The histories from each fold.
    """
    for i, history in enumerate(histories):
        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Fold {i + 1} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Fold {i + 1} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])

        plt.show()


def main():
    """
    Main function to prepare data, train and evaluate the hybrid recommender model, and generate recommendations.

    This function performs the following steps:
    1. Prepares the data for the model.
    2. Creates mappings for user and word IDs.
    3. Trains and evaluates the model using cross-validation.
    4. Connects to the database.
    5. Splits the data into training and testing sets.
    6. Trains the final model on the training set.
    7. Evaluates the recommendations on the test set.
    8. Generates and prints recommendations for a sample of users.
    """
    X, y, preprocessor = prepare_data_for_model()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    user_mapping = {user: i for i, user in enumerate(X['user_id'].unique())}
    word_mapping = {word: i for i, word in enumerate(X['mot_id'].unique())}

    fold_performances = train_and_evaluate_model(X, y, user_mapping, word_mapping, len(user_mapping), len(word_mapping))

    # Visualise loss and precision curves
    histories = [perf[2] for perf in fold_performances]
    plot_history(histories)

    mean_val_loss = np.mean([perf[0] for perf in fold_performances])
    mean_val_accuracy = np.mean([perf[1] for perf in fold_performances])

    print(f"\nMean Validation Loss: {mean_val_loss:.4f}")
    print(f"Mean Validation Accuracy: {mean_val_accuracy:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create user_id and mot_id mappings to indices
    user_mapping = {user: i for i, user in enumerate(X['user_id'].unique())}
    word_mapping = {word: i for i, word in enumerate(X['mot_id'].unique())}

    num_users = len(user_mapping)
    num_words = len(word_mapping)

    model = create_hybrid_recommender(num_users, num_words, X_train.shape[1] - 2)
    model.fit(
        [X_train['user_id'].map(user_mapping), X_train['mot_id'].map(word_mapping),
         X_train.drop(['user_id', 'mot_id'], axis=1)],
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    conn = get_db_connection()

    # Evaluate the recommendations
    metrics = evaluate_recommendations(model, X_test, y_test, user_mapping, word_mapping, conn)
    print("Métriques d'évaluation des recommandations:")
    print(metrics)

    # Save the evaluation metrics to a file
    evaluation_file_path = os.path.join(os.getcwd(), 'modele1_improved_evaluation.txt')
    with open(evaluation_file_path, 'w') as file:
        file.write("Métriques d'évaluation des recommandations:\n")
        for metric, value in metrics.items():
            file.write(f"{metric}: {value:.4f}\n")
    print(f"Les métriques d'évaluation ont été sauvegardées dans {evaluation_file_path}")

    # Example of predictions for some users
    sample_users = X_test['user_id'].unique()[:5]
    recommendations = predict_recommendations(model, X_test[X_test['user_id'].isin(sample_users)], user_mapping,
                                              word_mapping, conn)
    for user, recs in recommendations.items():
        print(f"\nRecommandations pour l'utilisateur {user}:")
        for rec in recs:
            print(f"  - Mot: {rec['mot']}")
            print(f"    ID: {rec['mot_id']}")
            print(f"    Niveau de difficulté: {rec['niveau_difficulte']}")
            print(f"    Catégorie: {rec['categorie']}")
            print(f"    Sous-catégorie: {rec['sous_categorie']}")
        print()


if __name__ == "__main__":
    main()
