"""
CamemBERT Processing Module

This module processes prepared text data using the CamemBERT model to generate embeddings
for each word in the input sentences.

Functions:
    process_sentence(sentence: str) -> List[Tuple[str, np.ndarray]]:
        Processes a single sentence using CamemBERT and returns word embeddings.

    process_all_sentences() -> None:
        Processes all sentences in the input file and saves the results.

Usage:
    Run this script to process the prepared data and generate word embeddings
    using the CamemBERT model.

Dependencies:
    - transformers
    - torch
    - numpy

Author: Marianne Arrué
Date: 22/08/24
"""

import torch
from transformers import CamembertTokenizer, CamembertModel
import numpy as np

# Configuration
INPUT_FILE = "prepared_data_for_camembert_test.txt"
OUTPUT_FILE = "camembert_processed_data_test.txt"
MAX_LENGTH = 128  # Longueur maximale de séquence pour CamemBERT

# Charger le tokenizer et le modèle CamemBERT
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")


def process_sentence(sentence):
    # Tokeniser la phrase
    inputs = tokenizer(sentence, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")

    # Obtenir les embeddings du modèle
    with torch.no_grad():
        outputs = model(**inputs)

    # Extraire les embeddings de la dernière couche cachée
    last_hidden_states = outputs.last_hidden_state[0]  # Shape: (sequence_length, hidden_size)

    # Convertir les tokens en mots
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    # Associer chaque mot à son embedding
    word_embeddings = []
    current_word = ""
    current_embedding = []

    for token, embedding in zip(tokens, last_hidden_states):
        if token.startswith("▁"):  # Nouveau mot
            if current_word:
                word_embeddings.append((current_word, torch.mean(torch.stack(current_embedding), dim=0).numpy()))
            current_word = token[1:]
            current_embedding = [embedding]
        elif token == "</s>":  # Fin de séquence
            break
        else:
            current_word += token
            current_embedding.append(embedding)

    # Ajouter le dernier mot
    if current_word:
        word_embeddings.append((current_word, torch.mean(torch.stack(current_embedding), dim=0).numpy()))

    return word_embeddings


def process_all_sentences():
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for line in infile:
            sentence = line.strip()
            if sentence:
                word_embeddings = process_sentence(sentence)
                for word, embedding in word_embeddings:
                    # Écrire le mot et une représentation de son embedding
                    outfile.write(f"{word}\t{','.join(map(str, embedding[:5]))}\n")
                outfile.write("\n")  # Séparateur entre les phrases

    print(f"Traitement terminé. Résultats sauvegardés dans {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_sentences()