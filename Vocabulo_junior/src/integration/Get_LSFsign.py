"""
LSF Sign Matching Module

This module processes the CamemBERT embeddings and matches words to their corresponding
LSF (French Sign Language) signs based on various linguistic features and database information.

Functions:
    connect_to_db() -> sqlalchemy.engine.base.Engine:
        Establishes a connection to the PostgreSQL database.

    map_pos_to_gramm_id(pos: str) -> int:
        Maps part-of-speech tags to grammatical IDs used in the database.

    get_word_info(word: str, lemma: str, pos: str, connection) -> List[Dict]:
        Retrieves word information from the database.

    get_sentence_embedding(sentence: str) -> np.ndarray:
        Generates a sentence embedding using CamemBERT.

    choose_best_definition(word: str, context: str, definitions: List[Dict], pos: str) -> Dict:
        Selects the most appropriate definition based on context and similarity.

    process_text_and_match_lsf(prepared_file: str, embedding_file: str) -> List[Dict]:
        Main function to process text and match words to LSF signs.

Usage:
    Run this script to process CamemBERT embeddings, analyze text, and match words
    to their corresponding LSF signs.

Dependencies:
    - sqlalchemy
    - transformers
    - torch
    - nlp
    - sklearn
    - numpy

Author: Marianne Arrué
Date: 02/09/24
"""
import os
import csv
import re
import numpy as np
import torch
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from transformers import CamembertModel, CamembertTokenizer, CamembertForTokenClassification
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode


# Load the French spaCy model
nlp = spacy.load("fr_core_news_md")

# Configuration
PREPARED_DATA_FILE = "prepared_data_for_camembert_test.txt"
EMBEDDING_DATA_FILE = "camembert_processed_data_test.txt"
OUTPUT_FILE = "words_with_lsf_signs_test2.csv"

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'BDD', '.env')
load_dotenv(dotenv_path)

# Database connection configuration
db_params = {
    'dbname': os.environ.get('POSTGRES_DB'),
    'user': os.environ.get('POSTGRES_USER'),
    'password': os.environ.get('POSTGRES_PASSWORD'),
    'host': 'localhost',
    'port': '5432'
}

# Load models
camembert_base = CamembertModel.from_pretrained('camembert-base')
camembert_base_tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
camembert_ner = CamembertForTokenClassification.from_pretrained('jean-baptiste/camembert-ner')
camembert_ner_tokenizer = CamembertTokenizer.from_pretrained('jean-baptiste/camembert-ner')


def connect_to_db():
    """
    Establish a connection to the PostgreSQL database.

    Returns:
    sqlalchemy.engine.base.Engine: Database engine connection.
    """
    try:
        conn_string = (f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:"
                       f"{db_params['port']}/{db_params['dbname']}")
        engine = create_engine(conn_string)
        print("Connexion à la base de données réussie.")
        return engine
    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        raise


def map_pos_to_gramm_id(pos):
    """
    Map part-of-speech tags to grammatical IDs used in the database.

    Parameters:
    pos (str): Part-of-speech tag.

    Returns:
    int: Corresponding grammatical ID.
    """
    mapping = {
        'NOUN': 1,  # noun
        'VERB': 2,  # verb
        'AUX': 2,  # auxiliary (treated as verb)
        'ADJ': 3,  # adjective
        'ADP': 4,  # preposition
        'DET': 5,  # determiner
        'PRON': 6,  # pronoun
        'PROPN': 7,  # proper noun
        'ADV': 8,  # adverb
        'CCONJ': 9,  # coordinating conjunction
        'SCONJ': 9,  # subordinating conjunction
        'INTJ': 10  # interjection
    }
    return mapping.get(pos, 1)  # Par défaut, retourner 1 (nom) si pas trouvé


def get_word_info(word, lemma, pos, connection):
    """
    Retrieve word information from the database.

    Parameters:
    word (str): The word to look up.
    lemma (str): The lemma of the word.
    pos (str): Part-of-speech tag.
    connection: Database connection.

    Returns:
    List[Dict]: List of word information dictionaries.
    """
    gramm_id = map_pos_to_gramm_id(pos)

    query = text("""
    SELECT m.mot_id, m.mot, m.definition, ls.url_sign, ls.url_def, gc.name as gram_cat
    FROM mot m
    LEFT JOIN lsf_signe ls ON m.mot_id = ls.mot_id
    LEFT JOIN grammatical_cat gc ON m.gramm_id = gc.gramm_id
    WHERE (LOWER(m.mot) = LOWER(:word) OR LOWER(m.mot) = LOWER(:lemma)) 
    AND m.gramm_id = :gramm_id
    AND m.definition IS NOT NULL AND m.definition != ''
    """)
    results = connection.execute(query, {"word": word, "lemma": lemma, "gramm_id": gramm_id}).fetchall()
    return [row._asdict() for row in results]


def get_sentence_embedding(sentence):
    """
    Generate a sentence embedding using CamemBERT.

    Parameters:
    sentence (str): The sentence to embed.

    Returns:
    np.ndarray: The sentence embedding.
    """
    inputs = camembert_base_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = camembert_base(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def disambiguate_word_sense(word, context, definitions):
    """
    Disambiguate the word sense based on context and definitions.

    Parameters:
    word (str): The word to disambiguate.
    context (str): The context in which the word appears.
    definitions (List[Dict]): List of possible definitions.

    Returns:
    Dict: The best matching definition.
    """
    # Utiliser un modèle de langue plus avancé pour comprendre le contexte
    context_embedding = get_sentence_embedding(context)

    best_score = -1
    best_definition = None

    for definition in definitions:
        def_embedding = get_sentence_embedding(definition['definition'])
        score = cosine_similarity([context_embedding], [def_embedding])[0][0]

        if score > best_score:
            best_score = score
            best_definition = definition

    return best_definition


def choose_best_definition(word, context, definitions, pos):
    """
    Select the most appropriate definition based on context and similarity.

    Parameters:
    word (str): The word to define.
    context (str): The context in which the word appears.
    definitions (List[Dict]): List of possible definitions.
    pos (str): Part-of-speech tag.

    Returns:
    Dict: The best matching definition.
    """
    if not definitions:
        return None
    if len(definitions) == 1:
        return definitions[0]

    context_embedding = get_sentence_embedding(context)

    scores = []
    for definition in definitions:
        def_embedding = get_sentence_embedding(definition['definition'])
        similarity = cosine_similarity([context_embedding], [def_embedding])[0][0]
        scores.append((definition, similarity))

    # Sort definitions by similarity score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nPolysemy analysis for the word '{word}' :")
    print(f"Context : {context}")
    print("\nSimilarity scores for each definition:")
    for definition, score in scores:
        print(f"Score : {score:.4f} - Définition : {definition['definition']}")

    best_definition, best_score = scores[0]
    print(f"\nBest definition chosen: : {best_definition['definition']}")
    print(f"Score : {best_score:.4f}")

    return best_definition


def get_pos_with_camembert(word, context):
    """
    Get the part-of-speech tag using CamemBERT.

    Parameters:
    word (str): The word to tag.
    context (str): The context in which the word appears.

    Returns:
    str: The part-of-speech tag.
    """
    inputs = camembert_ner_tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = camembert_ner(**inputs)

    word_tokens = camembert_ner_tokenizer.tokenize(word)
    context_tokens = camembert_ner_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    try:
        word_index = context_tokens.index(word_tokens[0])
    except ValueError:
        return "UNKNOWN"

    predicted_label = outputs.logits[0, word_index].argmax().item()
    pos = camembert_ner.config.id2label[predicted_label]

    return pos


def read_prepared_data(prepared_file, embedding_file):
    """
    Read prepared data and embeddings from files.

    Parameters:
    prepared_file (str): Path to the prepared data file.
    embedding_file (str): Path to the embedding data file.

    Returns:
    Tuple[List[str], Dict[str, Dict[str, str]]]: List of sentences and dictionary of embeddings.
    """
    with open(prepared_file, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()

    embeddings = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        current_sentence = []
        for line in f:
            line = line.strip()
            if line == '<s>' or line == '</s>':
                if current_sentence:
                    sentence = ' '.join(word for word, _ in current_sentence)
                    embeddings[sentence] = dict(current_sentence)
                    current_sentence = []
            elif line:
                parts = line.split('\t')
                if len(parts) == 2:
                    word, embedding = parts
                    current_sentence.append((word.strip(), embedding.strip()))

    return sentences, embeddings


def correct_lemma(word, lemma):
    """
    Correct the lemma of a word.

    Parameters:
    word (str): The original word.
    lemma (str): The lemma of the word.

    Returns:
    str: The corrected lemma.
    """
    # If the lemma is the same as the word but without accents, keep the original word
    if unidecode(word) == lemma:
        return word

    # List of French infinitive verb endings
    infinitive_endings = ['er', 'ir', 're']

    # If the lemma ends with one of these endings, keep it
    if any(lemma.endswith(ending) for ending in infinitive_endings):
        return lemma

    # Otherwise, check if the original word could be an infinitive
    if any(word.endswith(ending) for ending in infinitive_endings):
        return word

    # If none of these conditions are met, keep the original lemma
    return lemma


def analyze_sentence(sentence):
    """
    Analyze a sentence using spaCy.

    Parameters:
    sentence (str): The sentence to analyze.

    Returns:
    List[Dict]: List of analyzed word information.
    """
    doc = nlp(sentence)

    analyzed_words = []
    for token in doc:
        corrected_lemma = correct_lemma(token.text, token.lemma_)
        analyzed_words.append({
            'word': token.text,
            'lemma': corrected_lemma,
            'pos': token.pos_,
            'dep': token.dep_
        })

    has_subject = any(word['dep'] in ['nsubj', 'nsubjpass'] for word in analyzed_words)
    has_verb = any(word['pos'] == 'VERB' for word in analyzed_words)

    if not (has_subject and has_verb):
        print(f"Warning: The sentence '{sentence}' might be incomplete.")

    return analyzed_words


def analyze_context(sentence):
    """
    Extract rich context information from a sentence using spaCy.

    Parameters:
    sentence (str): The sentence to analyze.

    Returns:
    Dict: Dictionary containing context information.
    """
    doc = nlp(sentence)

    context_info = {
        'subject': [token.text for token in doc if token.dep_ == 'nsubj'],
        'verbs': [token.lemma_ for token in doc if token.pos_ == 'VERB'],
        'objects': [token.text for token in doc if token.dep_ in ['dobj', 'pobj']],
        'adjectives': [token.text for token in doc if token.pos_ == 'ADJ']
    }

    return context_info

def get_best_pos(word, context, spacy_pos):
    """
    Determine the best part-of-speech tag using both spaCy and CamemBERT.

    Parameters:
    word (str): The word to tag.
    context (str): The context in which the word appears.
    spacy_pos (str): The part-of-speech tag from spaCy.

    Returns:
    str: The best part-of-speech tag.
    """
    camembert_pos = get_pos_with_camembert(word, context)

    pos_priorities = {
        'NOUN': 3, 'VERB': 3, 'ADJ': 2, 'ADV': 2, 'PRON': 2,
        'DET': 1, 'ADP': 1, 'CONJ': 1, 'NUM': 1, 'PART': 1, 'INTJ': 1
    }

    spacy_priority = pos_priorities.get(spacy_pos, 0)
    camembert_priority = pos_priorities.get(camembert_pos, 0)

    return spacy_pos if spacy_priority >= camembert_priority else camembert_pos


def process_text_and_match_lsf(prepared_file, embedding_file):
    """
    Process text and match words to LSF signs.

    Parameters:
    prepared_file (str): Path to the prepared data file.
    embedding_file (str): Path to the embedding data file.

    Returns:
    List[Dict]: List of results with matched LSF signs.
    """
    sentences, embeddings = read_prepared_data(prepared_file, embedding_file)
    engine = connect_to_db()

    results = []
    with engine.connect() as connection:
        for sentence in sentences:
            sentence_embeddings = embeddings.get(sentence, {})
            analyzed_words = analyze_sentence(sentence)

            for word_info in analyzed_words:
                word = word_info['word']
                lemma = word_info['lemma']
                pos = word_info['pos']

                embedding = sentence_embeddings.get(word, None)

                word_data = get_word_info(word, lemma, pos, connection)

                if word_data:
                    best_match = choose_best_definition(word, sentence, word_data, pos)
                    if best_match:
                        url = best_match['url_sign'] or best_match['url_def'] or 'Pas de signe/définition URL'
                        results.append({
                            'sentence': sentence,
                            'word': word,
                            'lemma': lemma,
                            'pos': best_match['gram_cat'],  # Utiliser la catégorie de la BDD
                            'definition': best_match['definition'],
                            'url': url,
                            'embedding': embedding
                        })
                else:
                    results.append({
                        'sentence': sentence,
                        'word': word,
                        'lemma': lemma,
                        'pos': pos,
                        'definition': 'Non trouvé dans la BDD',
                        'url': 'Non disponible',
                        'embedding': embedding
                    })

    return results


if __name__ == "__main__":
    results = process_text_and_match_lsf(PREPARED_DATA_FILE, EMBEDDING_DATA_FILE)

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Phrase', 'Mot', 'Lemme', 'POS', 'Définition', 'URL Signe', 'Embedding'])
        for result in results:
            csv_writer.writerow([
                result['sentence'],
                result['word'],
                result['lemma'],
                result['pos'],
                result['definition'],
                result['url'],
                result['embedding']
            ])

    print(f"Processing complete. Results saved in {OUTPUT_FILE}")