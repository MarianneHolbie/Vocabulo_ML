"""
This script generates fake data for the Vocabulo_quiz application. It creates users, quizzes, and user word history
using the Faker library and inserts the data into a PostgreSQL database.

The script includes functions to:
- Generate device tokens for users.
- Generate users with authentication details.
- Retrieve available words from the database.
- Generate quiz data for users over a specified number of days.
- Generate user word history for existing users.
- Generate new users without quiz history.
"""

import random
from datetime import datetime, timedelta
import psycopg2
import uuid
from faker import Faker
import os
from dotenv import load_dotenv

fake = Faker('fr_FR')
load_dotenv()

try:
    conn = psycopg2.connect(
        dbname=os.environ.get('POSTGRES_DB'),
        user=os.environ.get('POSTGRES_USER'),
        password=os.environ.get('POSTGRES_PASSWORD'),
        host='localhost'
    )
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {str(e)}")
    raise

cur = conn.cursor()

def generate_device_tokens(num_tokens):
    """
    Generate device tokens for users.

    Parameters:
    num_tokens (int): Number of device tokens to generate.

    Returns:
    list: List of generated device token IDs.
    """
    token_ids = []
    for _ in range(num_tokens):
        cur.execute(
            "INSERT INTO device_token (create_at, device_token, device_type) VALUES (%s, %s, %s) RETURNING token_id",
            (fake.date_time_this_year(), fake.uuid4(), random.choice(['ios', 'android', 'web']))
        )
        token_ids.append(cur.fetchone()[0])
    conn.commit()
    return token_ids

def generate_users(num_users):
    """
    Generate users with authentication details.

    Parameters:
    num_users (int): Number of users to generate.

    Returns:
    list: List of generated user IDs.
    """

    token_ids = generate_device_tokens(num_users)
    user_ids = []
    for token_id in token_ids:
        user_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO authentication (user_id, pseudo, password, date, token_id) VALUES (%s, %s, %s, %s, %s) RETURNING user_id",
            (user_id, fake.user_name(), fake.password(), fake.date_time_this_year(), token_id)
        )
        user_ids.append(cur.fetchone()[0])
    conn.commit()
    return user_ids

def get_available_words():
    """
    Retrieve available words from the database.

    Returns:
    list: List of tuples containing word details.
    """
    cur.execute("""
        SELECT m.mot_id, m.niv_diff_id, m.gramm_id, 
               array_agg(DISTINCT mc.categorie_id) as categories
        FROM mot m 
        JOIN lsf_signe ls ON m.mot_id = ls.mot_id 
        LEFT JOIN mot_categorie mc ON m.mot_id = mc.mot_id
        WHERE ls.url_sign != 'Non spécifié' OR ls.url_def != 'Non spécifié'
        GROUP BY m.mot_id, m.niv_diff_id, m.gramm_id
    """)
    return cur.fetchall()

def generate_quiz_data(user_ids, num_days):
    """
     Generate quiz data for users over a specified number of days.

     Parameters:
     user_ids (list): List of user IDs.
     num_days (int): Number of days to generate quiz data for.
     """
    available_words = get_available_words()

    for user_id in user_ids:
        start_date = fake.date_time_this_year()

        for day in range(num_days):
            quiz_date = start_date + timedelta(days=day)
            quiz_id = str(uuid.uuid4())

            cur.execute(
                "INSERT INTO eval_quiz (eval_quizz_id, quiz_id, scale) VALUES (%s, %s, %s)",
                (str(uuid.uuid4()), quiz_id, random.choice(['Trop dur', 'Bien', 'Trop facile']))
            )

            cur.execute(
                "INSERT INTO quiz (quiz_id, date, user_id) VALUES (%s, %s, %s)",
                (quiz_id, quiz_date, user_id)
            )

            quiz_words = random.sample(available_words, random.randint(2, 5))

            for word in quiz_words:
                mot_id = word[0]
                score = random.choice([True, False])
                use_sign = random.choice([True, False])
                count = random.randint(0, 2)

                score_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO score_quiz (score_id, quiz_id, mot_id, score, use_sign, count) VALUES (%s, %s, %s, %s, %s, %s)",
                    (score_id, quiz_id, mot_id, score, use_sign, count)
                )

                cur.execute(
                    "INSERT INTO view_sign (score_id, flag) VALUES (%s, %s)",
                    (score_id, use_sign)
                )

                cur.execute(
                    "INSERT INTO eval_mot (eval_mot_id, quiz_id, mot_id, scale) VALUES (%s, %s, %s, %s)",
                    (str(uuid.uuid4()), quiz_id, mot_id, random.choice(['Trop dur', 'Bien', 'Trop facile']))
                )

        conn.commit()

def generate_user_word_history(user_ids, mot_ids):
    """
    Generate user word history for existing users.

    Parameters:
    user_ids (list): List of user IDs.
    mot_ids (list): List of word IDs.
    """
    for user_id in user_ids:
        processed_words = set()

        for mot_id in random.sample(mot_ids, k=min(len(mot_ids), 100)):
            if mot_id in processed_words:
                continue

            processed_words.add(mot_id)
            times_seen = random.randint(1, 10)
            times_correct = random.randint(0, times_seen)
            last_seen = fake.date_time_between(start_date="-30d", end_date="now")

            try:
                cur.execute("""
                    INSERT INTO user_word_history (user_id, mot_id, last_seen, times_seen, times_correct)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, mot_id) DO UPDATE
                    SET last_seen = EXCLUDED.last_seen,
                        times_seen = user_word_history.times_seen + EXCLUDED.times_seen,
                        times_correct = user_word_history.times_correct + EXCLUDED.times_correct
                """, (user_id, mot_id, last_seen, times_seen, times_correct))
            except Exception as e:
                print(f"Error inserting data for user {user_id} and word {mot_id}: {str(e)}")
                conn.rollback()
            else:
                conn.commit()

    print("User word history generation completed.")

def generate_users_without_history(num_users):
    """
    Generate new users without quiz history.

    Parameters:
    num_users (int): Number of new users to generate.

    Returns:
    list: List of generated user IDs.
    """
    user_ids = []
    for _ in range(num_users):
        user_id = str(uuid.uuid4())
        token_id = generate_device_tokens(1)[0]
        cur.execute(
            "INSERT INTO authentication (user_id, pseudo, password, date, token_id) VALUES (%s, %s, %s, %s, %s) RETURNING user_id",
            (user_id, fake.user_name(), fake.password(), fake.date_time_this_year(), token_id)
        )
        user_id = cur.fetchone()[0]
        user_ids.append(user_id)
        print(f"Generated user with ID: {user_id}")
    conn.commit()
    return user_ids

# Execution
num_existing_users = 200
num_days_for_quiz = 30
num_new_users = 10

# Generate existing users and their quiz data
existing_user_ids = generate_users(num_existing_users)
generate_quiz_data(existing_user_ids, num_days_for_quiz)

# Generate word history for existing users
cur.execute("SELECT mot_id FROM mot")
mot_ids = [row[0] for row in cur.fetchall()]
generate_user_word_history(existing_user_ids, mot_ids)

# Generate new users without history
new_user_ids = generate_users_without_history(num_new_users)

print(f"Generated {len(existing_user_ids)} existing users with quiz history")
print(f"Generated {len(new_user_ids)} new users without history")

cur.close()
conn.close()
print("Script completed successfully")