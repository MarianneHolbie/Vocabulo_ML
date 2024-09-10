"""
automate_scraptWord.py

This script automates the process of scraping definitions and related information for words from an online dictionary.

Usage:
    python automate_scraptWord.py

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
import time
import csv


def extract_word_info(url):
    """
    Fetches the HTML content of the dictionary page for the given word and extracts relevant information.

    Parameters:
    url (str): The URL of the dictionary page to scrape

    Returns:
    dict: A dictionary containing word information, or None if an error occurs
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        script = soup.find('script', string=re.compile('window.__PRELOADED_STATE__'))
        if not script:
            print(f"Impossible de trouver les données pour l'URL : {url}")
            return None

        json_text = re.search(r'window.__PRELOADED_STATE__ = (.+);', script.string)
        if not json_text:
            print(f"Unable to find data for URL: {url}")
            return None

        data = json.loads(json_text.group(1))

        if 'meaning' not in data or 'data' not in data['meaning']:
            print(f"Structure de données inattendue pour l'URL : {url}")
            return None

        word_info = data['meaning']['data']
        word = word_info.get('word', {}).get('name', 'Non spécifié')
        typologie = word_info.get('word', {}).get('typology', 'Non spécifié')
        definition = word_info.get('definition', 'Non spécifié')

        # Gestion plus sûre des URLs de vidéo
        defsign_url = 'Non spécifié'
        if word_info.get('definitionSigns') and len(word_info['definitionSigns']) > 0:
            defsign_url = word_info['definitionSigns'][0].get('uri', 'Non spécifié')

        wordsign_url = 'Non spécifié'
        if word_info.get('wordSigns') and len(word_info['wordSigns']) > 0:
            wordsign_url = word_info['wordSigns'][0].get('uri', 'Non spécifié')

        return {
            'mot': word,
            'categorie_grammaticale': typologie,
            'definition': definition,
            'url_video_definition': defsign_url,
            'url_video_mot': wordsign_url,
            'url_source': url
        }
    except requests.RequestException as e:
        print(f"Request error for URL {url}: {e}")
    except json.JSONDecodeError:
        print(f"JSON decoding error for URL {url}")
    except Exception as e:
        print(f"Unexpected error for URL {url}: {e}")

    return None


def process_urls_file(input_file, output_file):
    """
    Processes a file containing URLs, scrapes information for each URL, and saves the data to a CSV file.

    Parameters:
    input_file (str): The path to the input file containing URLs
    output_file (str): The path to the output CSV file

    Returns:
    None
    """
    with open(input_file, 'r', encoding='utf-8') as f, \
            open(output_file, 'w', newline='', encoding='utf-8') as csvfile:

        csv_writer = csv.DictWriter(csvfile,
                                    fieldnames=['mot', 'categorie_grammaticale', 'definition', 'url_video_definition',
                                                'url_video_mot', 'url_source'])
        csv_writer.writeheader()

        for line in f:
            if line.startswith('http'):
                url = line.strip()
                print(f"Processing URL: {url}")
                word_data = extract_word_info(url)

                if word_data:
                    csv_writer.writerow(word_data)
                else:
                    print(f"No data extracted for URL: {url}")


            time.sleep(1)   # Pause for one second between each request


if __name__ == "__main__":
    input_file = "elix_lsf_urls.txt"
    output_file = "elix_lsf_data.csv"

    process_urls_file(input_file, output_file)
    print(f"Extraction completed. Data has been saved to {output_file}")