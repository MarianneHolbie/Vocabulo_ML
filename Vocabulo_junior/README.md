# Vocabulo Junior

## Overview

Vocabulo Junior is an innovative educational application designed to help deaf children read children's books using image
recognition and natural language processing. 
The project combines optical character recognition (OCR), advanced natural language processing (NLP) techniques and a
user-friendly interface to create an engaging learning experience.

## Objective

The primary objective of Vocabulo Junior is to bridge the gap between written French and French Sign Language (LSF) for 
deaf children. By leveraging cutting-edge technology, we aim to make reading more accessible and enjoyable for deaf 
children, ultimately enhancing their literacy skills and overall learning experience.

## Project Structure

```
vocabulo-junior/
│   ├── README.md
│   ├── notebooks_md/
│   │   ├── 01_data_collection.md
│   │   ├── 02_OCR.md
│   │   ├── 03_CamemBERT.md
│   │   ├── 04_spacy.md
│   │   ├── 05_integration.md
│   │   └── Finetuning_OCR.svg
│   ├── src/
│   │   ├── ocr/
│   │   ├── nlp/
│   │   │   └── camembert/
│   │   └── integration/
│   ├── models/
│   │   └── ocr/
│   ├── reports/
│   │   ├── ocr_performance/
│   │   └── nlp_evaluation/
```

## Key Features

1. **Image-to-Text Conversion**: Utilizes advanced OCR techniques to extract text from images of children's books.

2. **Text Analysis and Processing**: Employs CamemBERT and spaCy models for comprehensive text analysis, including 
part-of-speech tagging, lemmatization, and syntactic parsing.

3. **Word Difficulty Assessment**: Evaluates the complexity of words based on predefined criteria and user proficiency
levels.

4. **LSF Sign Language Integration**: Links words to corresponding French Sign Language signs or definitions when
available.

5**User-Friendly API**: Offers a robust API in docker container for seamless integration with frontend applications,
making it easy to build interactive user interfaces.(cf Github: https://github.com/TessierV/vocabulo/tree/Marianne_dev/)

## Development Journey

1. **OCR Integration**: I began by exploring various OCR solutions, ultimately selecting and fine-tuning Tesseract 
for optimal performance with children's book fonts and layouts.

2. **NLP Model Selection and Fine-tuning**: I chose CamemBERT as my base model.

3. **spaCy Integration**: To enhance my linguistic analysis capabilities, I integrated spaCy and customized its French
language model to better handle children's vocabulary and sentence structures.

4. **Database Design and Implementation**: I use the same database Postgres SQL based on Elix Dico to build a 
comprehensive database schema to store word information, definitions, LSF signs, and associated metadata, 
optimizing for quick retrieval and scalability.

5. **API Development**: Using FastAPI, I developed a API that integrates all components of the system, providing a 
seamless interface for frontend applications into a [docker container](https://github.com/TessierV/vocabulo/tree/Marianne_dev/vocabulo_junior_ml).

6. **Testing and Optimization**: I conducted unit testing at each stage of development.
Performance reports and optimizations are documented in our repository.



## Future Directions

1. **Kid dictionary**: Implement a specific dictionary for children with more appropriate definitions.

2. **Expanded LSF Content**: Increase the database of LSF signs and video content to provide more comprehensive sign 
language support.

3. **Test at school**: carried out a test by a class of the use of the application in a supported school setting.

4. **Advanced Analytics**: Develop a dashboard for parents and educators to track a child's progress and identify areas
for improvement.

5. **Community Features**: Implement features that allow users to contribute LSF signs, alternative definitions, or book
recommendations, fostering a collaborative learning environment.

By continually refining our technology and expanding our features, I aim to make Vocabulo Junior a tool in promoting 
literacy and language acquisition for deaf children.