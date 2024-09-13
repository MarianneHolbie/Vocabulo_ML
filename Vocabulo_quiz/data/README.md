# Data Directory

## Overview

This directory contains scripts and resources for generating synthetic data for the Vocabulo_quiz application. 
The primary purpose of this data generation process is to simulate user interactions and create a robust dataset for 
training and testing my recommendation model.

## Why Generate Data?

Building an effective machine learning model requires a substantial amount of data. In real-world scenarios, this data typically 
comes from user interactions with the application. However, in the early stages of development or when access to a large user 
base is limited, synthetic data generation becomes crucial.

For the Vocabulo_quiz project, I have implemented a data generation process to:

1. Simulate user behavior and interactions
2. Create a diverse dataset representing various user profiles
3. Generate quiz histories and word learning patterns
4. Provide a foundation for model training and testing

## Contents

- `generate_fake_data.py`: The main script for generating synthetic data

## How It Works

The `generate_fake_data.py` script performs the following tasks:

1. Generates user profiles with authentication details
2. Creates quiz data for existing users over a specified period
3. Simulates user word history, including times seen, times correct, and last seen dates
4. Generates new users without quiz history to represent newcomers

The script uses the Faker library to create realistic-looking data and interacts with a PostgreSQL database to store the 
generated information.

## Usage

To generate synthetic data:

1. Ensure you have the required dependencies installed (psycopg2, Faker)
2. Set up your PostgreSQL database and update the connection details in the script or environment variables
3. Run the script:

```
python generate_fake_data.py
```

You can adjust the following parameters in the script to control the data generation:

- `num_existing_users`: Number of users with quiz history
- `num_days_for_quiz`: Number of days to generate quiz data for each user
- `num_new_users`: Number of new users without quiz history

## Important Notes

- This synthetic data is meant for development and testing purposes only

By using this data generation process, I can develop and refine my recommendation model without relying on a large 
existing user base, accelerating the development cycle of the Vocabulo_quiz application.