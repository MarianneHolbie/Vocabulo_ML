"""
scrap_Elix_words.py

This script scrapes words from the Elix LSF dictionary by iterating through each letter's pages.

Usage:
    python scrap_Elix_words.py

Prerequisites:
    - Install necessary packages: requests, beautifulsoup4
    - Ensure internet connection for scraping

Author: Marianne Arrué
Date: 20/08/24
"""

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://dico.elix-lsf.fr/index-alphabetique/"


def get_words_from_letter(letter, max_pages):
    """
    Fetches words from the Elix LSF dictionary for a given letter and number of pages.

    Parameters:
    letter (str): The letter to fetch words for
    max_pages (int): The maximum number of pages to scrape for the given letter

    Returns:
    list: A list of words scraped from the dictionary
    """
    words = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}{letter}/{page}"
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, "html.parser")
        word_list = soup.select_one("ul.words")
        if word_list:
            word_links = word_list.select("li > a")
            words.extend([link.text for link in word_links])
    return words


def get_all_words():
    """
     Fetches all words from the Elix LSF dictionary by iterating through each letter's pages.

     Returns:
     list: A list of all words scraped from the dictionary
     """
    all_words = []
    letter_pages = {
        "a": 56, "b": 33, "c": 72, "d": 42, "e": 24, "f": 26, "g": 21, "h": 16,
        "i": 25, "j": 6, "k": 3, "l": 25, "m": 39, "n": 12, "o": 13, "p": 62,
        "q": 4, "r": 36, "s": 56, "t": 33, "u": 4, "v": 16, "w": 2, "x": 1,
        "y": 1, "z": 3
    }

    for letter, max_pages in letter_pages.items():
        words = get_words_from_letter(letter, max_pages)
        all_words.extend(words)

    return all_words

# Save the word list to a file
words = get_all_words()
with open("mots.txt", "w", encoding="utf-8") as file:
    for word in words:
        file.write(word + "\n")

print(f"Total number of words: {len(words)}")
print(f"Some words from the list: {', '.join(words[:10])}")
