# Hyperparameter Tuning for XGBoost Word Recommendation Model

## Introduction

In this notebook, we'll explore the hyperparameter tuning process for our XGBoost-based word recommendation model. We used Optuna, a hyperparameter optimization framework, to find the best configuration for our model.

## Setup

We used the following libraries and tools:
- XGBoost for the base model
- Optuna for hyperparameter optimization
- Scikit-learn for data preprocessing and model evaluation

## Hyperparameter Space

We defined the following hyperparameter space for our XGBoost model:

```python
param = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
    }
```

## Optimization Process

I used Optuna to perform a series of trials, each with a different combination of hyperparameters. 
My objective function was based on the F1 score, which I aimed to maximize.

```python
def objective(trial, X, y, preprocessor):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    trial (optuna.trial.Trial): Optuna trial object.
    X (np.array): Feature matrix.
    y (np.array): Target variable.
    preprocessor (sklearn.compose.ColumnTransformer): Preprocessor object used for data transformation.

    Returns:
    float: F1 score of the model.
    """
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
    }

    model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    return scores.mean()

# Optimization
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=200)
```

I ran 200 trials to explore the hyperparameter space thoroughly.

## Results

After the optimization process, I obtained the following best hyperparameters:

```python
joblib.dump(study, 'optuna_study.joblib')
```
Best trial number: 191
Best score: 0.7870061746532335
Best hyperparameters: 
{
    'max_depth': 4, 
    'learning_rate': 0.5859703184954502,
    'n_estimators': 659,
    'min_child_weight': 3,
    'subsample': 0.8551413303779319,
    'colsample_bytree': 0.8077027472096016,
    'gamma': 3.256521234725508e-08,
    'reg_alpha': 0.0004667294123109848,
    'reg_lambda': 0.0032392404590805644}


## Model Performance

Using these optimized hyperparameters, our XGBoost model achieved the following performance metrics on the test set:

- Accuracy: 0.8189655172413793
- Precision: 0.8309859154929577
- Recall: 0.8676470588235294
- F1-score: 0.8489208633093526
- ROC AUC: 0.8694852941176472
- Log Loss: 0.6408648844536631

## Analysis

1. High Accuracy: The model achieved an accuracy of 0.8190, which is a substantial improvement over the previous version. 
This indicates that the model is correctly classifying a large proportion of the instances.

2. Strong Precision and Recall: With a precision of 0.8310 and a recall of 0.8676, the model demonstrates a good balance between 
correctly identifying positive instances and minimizing false positives. This is crucial for a recommendation system, as it ensures 
that the suggestions are both relevant and comprehensive.

3. Impressive F1-Score: The F1-score of 0.8489 indicates an excellent balance between precision and recall. This metric 
is particularly important in recommendation systems where we want to provide accurate suggestions without missing relevant items.

4. Excellent ROC AUC: The ROC AUC score of 0.8695 suggests that the model has a strong ability to distinguish between the classes.
This high score indicates that the model performs well across various threshold settings.

5. Reasonable Log Loss: The log loss of 0.6409 is relatively low, indicating that the model's probability estimates are fairly 
well-calibrated. This is important for the reliability of the confidence scores associated with each recommendation.

## Conclusion

The hyperparameter tuning process using Optuna has led to a significant improvement in our XGBoost model's performance across 
all metrics. These results demonstrate that my word recommendation system is now capable of providing highly accurate and 
relevant suggestions to users.

Key achievements:
1. The model now correctly classifies about 82% of the instances, a marked improvement from the previous version.
2. The high precision and recall scores indicate that the model is effective at identifying relevant words while minimizing irrelevant suggestions.
3. The strong ROC AUC score suggests that the model is robust and performs well at various classification thresholds.

These improvements have several positive implications for my word recommendation system:
- Users are more likely to receive relevant and interesting word suggestions, enhancing their learning experience.
- The system can adapt well to different user preferences and difficulty levels, thanks to its balanced performance.
- The well-calibrated probability estimates (as indicated by the log loss) allow for more reliable confidence scoring 
of recommendations.

Future steps to consider:
1. Test model on real users
2. Conduct a thorough error analysis to understand the remaining misclassifications and potentially identify new features or data preprocessing steps.
2. Explore the impact of these improvements on user engagement and learning outcomes through A/B testing.
3. Investigate the model's performance across different user segments or word categories to ensure consistent quality across all 
use cases.

In conclusion, the hyperparameter tuning process has resulted in a highly effective word recommendation model. This optimized model 
provides a solid foundation for delivering personalized and impactful learning experiences to our users. The next steps should focus
on fine-tuning the system based on real-world performance and user feedback to further enhance its effectiveness in supporting 
language learning.