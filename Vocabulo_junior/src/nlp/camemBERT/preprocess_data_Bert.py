"""
Preprocessing Module for CamemBERT Input

This module prepares text data extracted from OCR for processing with the CamemBERT model.
It cleans the text, splits it into sentences, and formats it appropriately for CamemBERT input.

Functions:
    clean_text(text: str) -> str:
        Cleans the input text by removing special characters and normalizing spaces.

    process_file(file_path: str) -> str:
        Reads and cleans the content of a file.

    split_into_sentences(text: str) -> List[str]:
        Splits the text into sentences using NLTK's sentence tokenizer.

    prepare_data() -> None:
        Main function to prepare the data for CamemBERT processing.

Usage:
    Run this script to preprocess OCR-extracted text files and generate a single output file
    suitable for input into the CamemBERT model.

Author: Marianne Arrué
Date: 21/08/24
"""
import os
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary resources for NLTK
nltk.download('punkt_tab')

# Configuration
INPUT_FOLDER = "../OCR/ExtractedTexts_Img_test"
OUTPUT_FILE = "prepared_data_for_camembert_test.txt"

def clean_text(text):
    """
    Clean the text by removing special characters and normalizing spaces.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Remove special characters except basic punctuation
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s.,!?\''-]", '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(file_path):
    """
    Read and clean the content of a file.

    Parameters:
    file_path (str): The path to the file to be processed.

    Returns:
    str: The cleaned content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return clean_text(content)

def split_into_sentences(text):
    """
    Split the text into sentences.

    Parameters:
    text (str): The text to be split into sentences.

    Returns:
    List[str]: A list of sentences.
    """
    return sent_tokenize(text, language='french')

def prepare_data():
    """
    Prepare the data for CamemBERT processing.

    This function reads text files from the input folder, cleans and splits the text into sentences,
    and writes the processed sentences to the output file.
    """
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(INPUT_FOLDER):
            if filename.endswith('.txt'):
                file_path = os.path.join(INPUT_FOLDER, filename)
                cleaned_text = process_file(file_path)
                sentences = split_into_sentences(cleaned_text)
                for sentence in sentences:
                    outfile.write(sentence + '\n')
                outfile.write('\n')  # Ligne vide pour séparer les paragraphes

    print(f"Data prepared and saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()