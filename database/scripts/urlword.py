"""
urlword.py

This script scrapes URLs of dictionary pages for words from the Elix LSF dictionary sitemap.

Usage:
    python urlword.py

Prerequisites:
    - Install necessary packages: requests, beautifulsoup4
    - Ensure internet connection for scraping

Author: Marianne Arrué
Date: 20/08/24
"""

import requests
from bs4 import BeautifulSoup



def get_letter_sitemaps(main_sitemap_url):
    """
    Fetches the sitemap URLs for each letter from the main sitemap.

    Parameters:
    main_sitemap_url (str): The URL of the main sitemap

    Returns:
    dict: A dictionary where keys are letters and values are the corresponding sitemap URLs
    """
    response = requests.get(main_sitemap_url)
    soup = BeautifulSoup(response.content, 'xml')
    letter_sitemaps = {}

    for loc in soup.find_all('loc'):
        url = loc.text
        if 'sitemap-lettre-' in url:
            letter = url.split('-')[-1].split('.')[0]
            letter_sitemaps[letter] = url

    return letter_sitemaps


def get_word_urls(letter_sitemap_url):
    """
    Fetches the word URLs from a letter-specific sitemap.

    Parameters:
    letter_sitemap_url (str): The URL of the letter-specific sitemap

    Returns:
    list: A list of word URLs
    """
    response = requests.get(letter_sitemap_url)
    soup = BeautifulSoup(response.content, 'xml')
    word_urls = []

    for loc in soup.find_all('loc'):
        url = loc.text
        if '/dictionnaire/' in url:
            word_urls.append(url)

    return word_urls


def scrape_elix_lsf():
    """
    Scrapes all word URLs from the Elix LSF dictionary by iterating through each letter's sitemap.

    Returns:
    dict: A dictionary where keys are letters and values are lists of word URLs
    """
    main_sitemap_url = "https://dico.elix-lsf.fr/sitemap.xml"
    letter_sitemaps = get_letter_sitemaps(main_sitemap_url)

    all_word_urls = {}

    for letter, sitemap_url in letter_sitemaps.items():
        print(f"Récupération des URLs pour la lettre {letter}...")
        word_urls = get_word_urls(sitemap_url)
        all_word_urls[letter] = word_urls
        print(f"Nombre d'URLs trouvées pour la lettre {letter}: {len(word_urls)}")

    return all_word_urls


def save_urls_to_file(all_word_urls, filename="elix_lsf_urls.txt"):
    """
    Saves the scraped word URLs to a file.

    Parameters:
    all_word_urls (dict): A dictionary where keys are letters and values are lists of word URLs
    filename (str): The name of the file to save the URLs

    Returns:
    None
    """
    with open(filename, "w", encoding="utf-8") as f:
        for letter, urls in all_word_urls.items():
            f.write(f"Lettre {letter}:\n")
            for url in urls:
                f.write(f"{url}\n")
            f.write("\n")
    print(f"URLs have been saved to the file {filename}")


if __name__ == "__main__":
    all_word_urls = scrape_elix_lsf()
    save_urls_to_file(all_word_urls)

    total_urls = sum(len(urls) for urls in all_word_urls.values())
    print(f"Total number of URLs fetched:  {total_urls}")