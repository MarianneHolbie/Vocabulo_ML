import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from Vocabulo_quiz.src.db_connexion.database import get_db_connection


def load_data(engine):
    """
    Load relevant data from the database using the provided SQLAlchemy engine.

    Parameters:
    engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.

    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    query = """
    SELECT 
        sq.score, 
        q.user_id,
        m.mot_id, m.mot, m.niv_diff_id, m.frequence, m.gramm_id,
        c.name as category, 
        sc.name as subcategory,
        e.echelon_id,
        COALESCE(uwh.times_correct, 0) as times_correct,
        COALESCE(uwh.times_seen, 0) as times_seen,
        COALESCE(uwh.last_seen, '1970-01-01'::timestamp) as last_seen,
        diff.freqfilms, diff.freqlivres, diff.nbr_syll, diff.cp_cm2_sfi,
        q.date as quiz_date
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
    df = pd.read_sql(query, engine)
    return df


def explore_data(df):
    """
     Perform exploratory data analysis on the DataFrame.

     This function performs the following steps:
     1. Prints DataFrame information and basic statistics.
     2. Plots the distribution of scores.
     3. Plots the correlation matrix of numerical variables.
     4. Plots the distribution of categories.
     5. Plots the distribution of word frequencies in films and books.
     6. Plots the relationship between the number of syllables and difficulty level.
     7. Plots the correlation between different difficulty measures.
     8. Plots the distribution of words with specified sign URLs.

     Parameters:
     df (pd.DataFrame): DataFrame containing the data to be analyzed.
     """
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nBasic statistics:\n", df.describe())

    # Plot the distribution of scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['score'], kde=True)
    plt.title('Distribution des Scores')
    plt.savefig('score_distribution.png')
    plt.close()

    # Plot the correlation matrix of numerical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matrice de Corrélation')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Plot the distribution of categories
    plt.figure(figsize=(12, 6))
    df['category'].value_counts().plot(kind='bar')
    plt.title('Distribution des Catégories')
    plt.xticks(rotation=45)
    plt.savefig('category_distribution.png')
    plt.close()

    # Plot the distribution of word frequencies in films and books
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df['freqfilms'], color='blue', linestyle='-', label='Films')
    sns.kdeplot(data=df['freqlivres'], color='orange', linestyle='--', label='Livres')
    plt.title('Distribution des fréquences dans les films et les livres (KDE)')
    plt.xlabel('Fréquence')
    plt.legend()
    plt.show()

    # Plot the relationship between the number of syllables and difficulty level
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='nbr_syll', y='niv_diff_id')
    plt.title('Relation entre le nombre de syllabes et la difficulté')
    plt.xlabel('Nombre de syllabes')
    plt.ylabel('Niveau de difficulté')
    plt.show()

    # Plot the correlation between different difficulty measures
    correlation_matrix = df[['niv_diff_id', 'frequence', 'freqfilms', 'freqlivres', 'nbr_syll']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Corrélation entre les mesures de difficulté')
    plt.show()


if __name__ == "__main__":
    engine = get_db_connection()
    df = load_data(engine)
    explore_data(df)