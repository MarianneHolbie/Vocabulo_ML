import joblib
import os


def load_optuna_study(filename):
    """
    Load and print the contents of an Optuna study saved as a .joblib file.

    Parameters:
    filename (str): Path to the .joblib file containing the Optuna study.

    Returns:
    None
    """
    if os.path.exists(filename):
        # Load the study from the joblib file
        study = joblib.load(filename)

        # Print the best trial
        best_trial = study.best_trial
        print(f'Best trial number: {best_trial.number}')
        print(f'Best score: {best_trial.value}')
        print(f'Best hyperparameters: {best_trial.params}')

        # Optionally, print all trials
        print("\nAll Trials:")
        for trial in study.trials:
            print(f'Trial number: {trial.number}, Value: {trial.value}, Params: {trial.params}')
    else:
        print(f"File {filename} does not exist.")


# Specify the file path for the saved Optuna study
filename = "./optuna_study_final.joblib"

# Load and print the study
load_optuna_study(filename)
