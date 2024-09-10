"""
ScraptOneWord.py

This script scrapes definitions and related information for a specific word from an online dictionary.

Usage:
    python ScraptOneWord.py <word_to_scrape>

Prerequisites:
    - Install necessary packages: requests, beautifulsoup4
    - Ensure internet connection for scraping

Author: Marianne Arrué
Date: 20/08/24
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import csv


def extract_word_info(url):
    """
    Fetches the HTML content of the dictionary page for the given word.

    Parameters:
    url (str): The URL of the dictionary page to scrape

    Returns:
    list: A list of dictionaries containing word information, or None if an error occurs
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the JSON data in the page script
        script = soup.find('script', string=re.compile('window.__PRELOADED_STATE__'))
        if not script:
            print(f"Unable to find data for URL : {url}")
            return None

        # Extract JSON text
        json_text = re.search(r'window.__PRELOADED_STATE__ = (.+);', script.string)
        if not json_text:
            print(f"Unexpected data format for the URL : {url}")
            return None

        # Loading JSON data
        data = json.loads(json_text.group(1))
        print(data)
        if not data:
            print(f"The extracted JSON is empty or incorrect for the URL : {url}")
            return None

        # Extracting relevant information from JSON
        search_data = data.get('search', {}).get('results', [])
        if not search_data:
            print(f"No data found in the ‘results’ section for the URL : {url}")
            return None

        # Retrieve all definitions and videos
        word_infos = []
        for result in data['search']['results']:
            mot = result.get('name', 'Non spécifié')
            categorie_grammaticale = result.get('typology', 'Non spécifié')

            for meaning in result.get('meanings', []):
                definition = meaning.get('definition', 'Non spécifié')

                # URL for definition in LSF
                url_video_definition = 'Non spécifié'
                if meaning.get('definitionSigns') and len(meaning['definitionSigns']) > 0:
                    url_video_definition = meaning['definitionSigns'][0].get('uri', 'Non spécifié')

                # URL for LSF sign
                url_video_mot = 'Non spécifié'
                if meaning.get('wordSigns') and len(meaning['wordSigns']) > 0:
                    url_video_mot = meaning['wordSigns'][0].get('uri', 'Non spécifié')

                word_infos.append({
                    'mot': mot,
                    'categorie_grammaticale': categorie_grammaticale,
                    'definition': definition,
                    'url_video_definition': url_video_definition,
                    'url_video_mot': url_video_mot,
                    'url_source': url
                })

        return word_infos

    except requests.RequestException as e:
        print(f"Error when requesting URL {url}: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON for URL  {url}")
    except Exception as e:
        print(f"Unexpected error when processing the URL  {url}: {e}")

    return None


def get_elix_url_for_word(word):
    """
    Constructs the URL for the given word in the Elix dictionary.

    Parameters:
    word (str): The word to look up

    Returns:
    str: The constructed URL
    """

    base_url = "https://dico.elix-lsf.fr/dictionnaire/"
    return f"{base_url}{word}"


def process_word(word):
    """
    Processes the given word by scraping its information from the Elix dictionary.

    Parameters:
    word (str): The word to process

    Returns:
    list: A list of dictionaries containing word information, or None if no data is found
    """
    url = get_elix_url_for_word(word)
    print(f"Traitement du mot : {word} avec l'URL : {url}")

    word_data = extract_word_info(url)

    if word_data:
        return word_data
    else:
        return None


def save_word_data_to_csv(word_infos, output_file):
    """
    Saves the word information to a CSV file.

    Parameters:
    word_infos (list): A list of dictionaries containing word information
    output_file (str): The path to the output CSV file
    """
    fieldnames = ['mot', 'categorie_grammaticale', 'definition', 'url_video_definition', 'url_video_mot', 'url_source']

    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for word_info in word_infos:
            writer.writerow(word_info)



if __name__ == "__main__":
    # Take the user's input word
    word = input("Veuillez entrer le mot pour lequel vous souhaitez récupérer des informations : ")

    # CSV Output file
    output_file = "../BDD/data/elix_lsf_data_single.csv"

    # Processing the word and recovering the data
    result = process_word(word)

    # If data has been recovered, add it to the CSV file
    if result:
        save_word_data_to_csv(result, output_file)
    else:
        print(f"No data found for the word '{word}'.")