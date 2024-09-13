# Vocabulo ML Project

## Overview

Vocabulo ML is a comprehensive machine learning project aimed at enhancing language learning experiences for the deaf 
community. It encompasses two main applications: Vocabulo Quiz and Vocabulo Junior, each leveraging different ML 
techniques to provide interactive and personalized learning experiences.

### [Vocabulo Quiz](./Vocabulo_quiz)

Vocabulo Quiz is an adaptive vocabulary learning application that uses machine learning to personalize word 
recommendations based on user performance and learning patterns. It employs techniques such as:

- XGBoost for word difficulty prediction
- Collaborative filtering for personalized recommendations
- Natural Language Processing (NLP) for word relationship analysis

### [Vocabulo Junior](./Vocabulo_junior)

Vocabulo Junior is an innovative application designed to help deaf children learn vocabulary through image recognition
and natural language processing. It utilizes:

- Optical Character Recognition (OCR) for text extraction from images
- CamemBERT, a state-of-the-art French language model, for contextual word understanding
- spaCy for additional NLP tasks such as named entity recognition and part-of-speech tagging

## Project Structure

The project is organized into three main components:

1. **Database**: A shared PostgreSQL database containing vocabulary data, user information, and learning progress.
2. **Vocabulo Quiz ML**: Machine learning models and APIs for the Quiz application.
3. **Vocabulo Junior ML**: OCR and NLP models and APIs for the Junior application.

## Technical Architecture

We've adopted a microservices architecture using Docker containers:

- **Database Container**: A dedicated PostgreSQL container shared between both applications.
- **Vocabulo Quiz API Container**: Contains the ML models and API for the Quiz application.
- **Vocabulo Junior API Container**: Houses the OCR, CamemBERT, and spaCy models along with the API for the Junior 
application.

All containers are deployed on the same Docker network, allowing secure inter-container communication while maintaining
separation of concerns.

## Development Journey

This repository documents our journey in developing and refining the ML models for both applications. Key milestones include:

1. Initial data collection and preprocessing
2. Experimentation with various ML algorithms
3. Fine-tuning of models for French language specifics
4. Integration of sign language considerations
5. Continuous improvement based on user feedback and performance metrics

## Getting Started

This repository is not intended to be cloned and used as it does not contain the data for the database.
It does, however, provide a better understanding of the design process for the various parts of this project.

## Documentation

Detailed documentation for each component can be found in the `docs/` directory. This includes:

- API specifications
- Model architectures
- Database schema and relationships
- Deployment guides

## Author

Marianne Arrué 

