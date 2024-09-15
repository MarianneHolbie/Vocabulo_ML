# Data collection for Vocabulo Junior

## Overview
In order to test the OCRs under real conditions, I started by creating a dataset of text images from children's books 
from various sources:

- `L'École des Loisirs` (a French publishing house) sent me 16 books in PDF format to include in my dataset
- I retrieved public domain books from the internet to quickly increase the volume of data
- And I personally took photos of about 30 children's books with my phone.

## PROCESS
Children's books have the particularity of using numerous fonts, lacking homogeneity in the fonts used, and sometimes 
even the font size changes within a sentence (sometimes even the font itself). The backgrounds can be white or very 
colorful when the text is over illustrations... For an OCR, this is quite a challenge, as they are generally trained 
on white texts with classic fonts.

In the collected dataset, I paid attention to preserving these different representations in order to obtain results 
in most cases.
Dataset containing 452 images.

### Preprocessing:
For the first test, I simply converted the PDFs into PNGs for each page, generating a corpus of PNG and native HEIF images.

In a second step, I resized the images to focus only on the text sections, reducing character recognition within the 
illustrations and avoiding the pitfall of having to recognize multiple paragraphs on the same page.


