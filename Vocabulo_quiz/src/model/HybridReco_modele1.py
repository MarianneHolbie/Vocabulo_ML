import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Vocabulo_quiz.src.feature_engineering.data_prep_initial import prepare_data_for_model


def create_hybrid_recommender(num_users, num_words, embedding_size=32, user_features_dim=10, word_features_dim=15):
    """
    Create a hybrid recommender model using user and word embeddings along with additional features.

    Parameters:
    num_users (int): Number of unique users.
    num_words (int): Number of unique words.
    embedding_size (int): Size of the embedding vectors.
    user_features_dim (int): Dimension of user features.
    word_features_dim (int): Dimension of word features.

    Returns:
    Model: Compiled Keras model.
    """
    # Inputs
    user_input = Input(shape=(1,), name='user_input')
    word_input = Input(shape=(1,), name='word_input')
    user_features_input = Input(shape=(user_features_dim,), name='user_features_input')
    word_features_input = Input(shape=(word_features_dim,), name='word_features_input')

    # Embeddings
    user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
    word_embedding = Embedding(num_words, embedding_size, name='word_embedding')(word_input)

    # Flatten embeddings
    user_embedding_flat = Flatten()(user_embedding)
    word_embedding_flat = Flatten()(word_embedding)

    # Concatenate all features
    concatenated = Concatenate()([user_embedding_flat, word_embedding_flat, user_features_input, word_features_input])

    # Dense layers
    dense1 = Dense(64, activation='relu')(concatenated)
    dense2 = Dense(32, activation='relu')(dense1)

    # Output
    output = Dense(1, activation='sigmoid', name='output')(dense2)

    model = Model(inputs=[user_input, word_input, user_features_input, word_features_input], outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, X_user, X_word, X_user_features, X_word_features, y, epochs=10, batch_size=32):
    """
    Train the hybrid recommender model.

    Parameters:
    model (Model): Compiled Keras model.
    X_user (np.array): Array of user IDs.
    X_word (np.array): Array of word IDs.
    X_user_features (np.array): Array of user features.
    X_word_features (np.array): Array of word features.
    y (np.array): Array of target values.
    epochs (int): Number of epochs to train the model.
    batch_size (int): Batch size for training.

    Returns:
    History: Keras training history object.
    """
    history = model.fit(
        [X_user, X_word, X_user_features, X_word_features],
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return history


def evaluate_model(model, X_user_test, X_word_test, X_user_features_test, X_word_features_test, y_test, save_path='./model1_evaluation.txt'):
    """
     Evaluate the hybrid recommender model and save the results.

     Parameters:
     model (Model): Trained Keras model.
     X_user_test (np.array): Array of user IDs for testing.
     X_word_test (np.array): Array of word IDs for testing.
     X_user_features_test (np.array): Array of user features for testing.
     X_word_features_test (np.array): Array of word features for testing.
     y_test (np.array): Array of target values for testing.
     save_path (str): Path to save the evaluation results.

     Returns:
     None
     """
    loss, accuracy = model.evaluate([X_user_test, X_word_test, X_user_features_test, X_word_features_test], y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Save evaluation results to a file
    with open(save_path, 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {accuracy}\n")


def main():
    """
    Main function to prepare data, create, train, and evaluate the hybrid recommender model.

    Returns:
    None
    """
    # Prepare data
    X, y, user_features, word_features = prepare_data_for_model()

    # Create mappings for users and words
    user_mapping = {user: i for i, user in enumerate(X['user_id'].unique())}
    word_mapping = {word: i for i, word in enumerate(X['mot_id'].unique())}

    # Prepare inputs for the model
    X_user = np.array([user_mapping[user] for user in X['user_id']])
    X_word = np.array([word_mapping[word] for word in X['mot_id']])

    # Align user features with X
    user_feature_cols = [col for col in X.columns if col.startswith('user_') and col != 'user_id']
    X_user_features = X[user_feature_cols].values

    # Align word features with X
    word_feature_cols = [col for col in X.columns if col.startswith('word_') and col != 'mot_id']
    X_word_features = X[word_feature_cols].values

    print(f"X_user shape: {X_user.shape}")
    print(f"X_word shape: {X_word.shape}")
    print(f"X_user_features shape: {X_user_features.shape}")
    print(f"X_word_features shape: {X_word_features.shape}")
    print(f"y shape: {y.shape}")

    # Split data into training and testing sets
    X_user_train, X_user_test, X_word_train, X_word_test, X_user_features_train, X_user_features_test, X_word_features_train, X_word_features_test, y_train, y_test = train_test_split(
        X_user, X_word, X_user_features, X_word_features, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_hybrid_recommender(len(user_mapping), len(word_mapping),
                                      user_features_dim=X_user_features.shape[1],
                                      word_features_dim=X_word_features.shape[1])
    history = train_model(model, X_user_train, X_word_train, X_user_features_train, X_word_features_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_user_test, X_word_test, X_user_features_test, X_word_features_test, y_test)


if __name__ == "__main__":
    main()
