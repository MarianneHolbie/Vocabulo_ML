# Vocabulo Quiz

## Project Overview
Vocabulo Quiz is an innovative language learning application designed to enhance French vocabulary acquisition for the 
deaf community. The core of this application is a sophisticated hybrid recommendation system that personalizes the 
learning experience for each user.

## Objective
The primary goal of the recommendation system is to suggest words that are optimally challenging for each user, 
balancing between reinforcing known vocabulary and introducing new words. This approach aims to maximize learning 
efficiency and user engagement.

## Key Features
- Personalized word recommendations based on user performance and learning history
- Integration with the Elix dictionary for comprehensive word and LSF (French Sign Language) sign coverage
- Adaptive difficulty scaling to match user progress
- Diverse word selection to ensure a well-rounded vocabulary development

## Development Journey
The development of the Vocabulo Quiz recommendation system progressed through several key stages:
1. Initial data exploration and feature engineering
2. Baseline model development with simple heuristics
3. Implementation of machine learning models (Random Forest, XGBoost)
4. Advanced hybrid model incorporating collaborative and content-based filtering techniques
5. Hyperparameter optimization using Optuna
6. Continuous refinement based on simulated user interactions and performance metrics

The current state-of-the-art model, HybridReco_final, represents the culmination of this iterative development process, 
offering superior personalization and recommendation accuracy.

## Repository Structure
- `/data`: process to generate fake data for user
- `/notebooks_md`: Jupyter notebooks or markdown file detailing data analysis and model development steps
- `/reports`: Performance reports, visualizations, and model comparisons
- `/src`: Source code for data processing, feature engineering, and model implementation



## Future Directions
For the moment, the model has only been trained on simulated user data and dummy quizzes. We need to be able to create
a community of testers to try and use the model in real conditions and re-train it regularly on new data so that it 
is as close as possible to reality.