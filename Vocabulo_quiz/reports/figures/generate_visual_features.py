"""
This script generates visualizations for the processed data used in the Vocabulo_quiz application. It includes functions to:
- Explore the processed data and print basic statistics.
- Visualize the distribution of the target variable.
- Analyze and visualize the distribution of important numeric features.
- Calculate and visualize feature importance based on correlation with the target variable.
- Analyze and visualize user and word features.

The script uses the following libraries:
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- matplotlib: For creating static, animated, and interactive visualizations.
- seaborn: For making statistical graphics.

Functions:
- explore_processed_data: Explores the processed data and generates various visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Vocabulo_quiz.src.feature_engineering.data_prep_initial import prepare_data_for_model


def explore_processed_data(X, y, user_features, word_features):
    """
    Explores the processed data and generates various visualizations.

    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable.
    user_features (pd.DataFrame): DataFrame containing user features.
    word_features (pd.DataFrame): DataFrame containing word features.

    Returns:
    pd.DataFrame: DataFrame containing feature importance based on correlation with the target variable.
    """
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Shape of user_features:", user_features.shape)
    print("Shape of word_features:", word_features.shape)

    print("\nFeature names in X:", X.columns.tolist())

    print("\nBasic statistics of X (numeric features only):")
    print(X.select_dtypes(include=[np.number]).describe())

    print("\nDistribution of y:")
    print(y.value_counts(normalize=True))

    # Visualize the distribution of y
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Target Variable')
    plt.savefig('target_distribution.png')
    plt.close()

    # Analyze numeric features in X
    numeric_features = X.select_dtypes(include=[np.number]).columns

    # Distribution of important numeric features in X
    important_features = ['avg_score', 'avg_difficulty', 'frequency']
    fig, axes = plt.subplots(len(important_features), 1, figsize=(10, 4 * len(important_features)))
    for i, feature in enumerate(important_features):
        if feature in X.columns:
            sns.histplot(X[feature], ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('important_features_distribution_X.png')
    plt.close()

    # Analyze feature importance
    numeric_X = X.select_dtypes(include=[np.number])
    correlations = numeric_X.corrwith(y)
    feature_importance = pd.DataFrame({
        'feature': correlations.index,
        'importance': correlations.abs().values
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features (based on correlation with target)')
    plt.savefig('feature_importance.png')
    plt.close()

    print("\nTop 20 most important features (based on correlation with target):")
    print(feature_importance.head(20))

    # Analyze user features
    print("\nUser features:")
    print(user_features.describe())

    # Analyze word features
    print("\nWord features:")
    print(word_features.describe())

    # Visualize distributions of main word features
    user_numeric_features = user_features.select_dtypes(include=[np.number]).columns
    columns_to_exclude = ['quiz_id', 'token_id', 'quiz_count']
    user_numeric_features = user_numeric_features.difference(columns_to_exclude)

    fig, axes = plt.subplots(len(user_numeric_features), 1, figsize=(10, 4 * len(user_numeric_features)))
    for i, feature in enumerate(user_numeric_features):
        sns.histplot(user_features[feature], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('user_features_distribution.png')
    plt.close()

    # Visualisation des distributions des principales caractéristiques des mots
    word_numeric_features = word_features.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(word_numeric_features), 1, figsize=(10, 4 * len(word_numeric_features)))
    for i, feature in enumerate(word_numeric_features):
        sns.histplot(word_features[feature], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('word_features_distribution.png')
    plt.close()

    return feature_importance


if __name__ == "__main__":
    X, y, user_features, word_features = prepare_data_for_model()
    feature_importance = explore_processed_data(X, y, user_features, word_features)