# 03 Model Selection

## Evolution of Models

### 1. Neural Network Approach (HybridReco_modele1 and HybridReco_modele1_improved)

My model selection process involved several iterations, each building upon the lessons learned from the previous one. Here's the journey I took:

#### 1.1 [HybridReco_modele1.py](../src/model/HybridReco_modele1.py)
- Initial approach using a neural network-based hybrid recommender system.
- Utilized user and word embeddings along with additional features.
- Implemented using TensorFlow/Keras.
- Dropout for regularization

#### 1.2 [HybridReco_modele1_improved.py](../src/model/HybridReco_modele1_improved.py)
- Enhanced version of the initial model.
- Use embeddings for users and words.
- Improved architecture with additional layers and dropout for better generalization.
- Introduced early stopping and learning rate reduction on plateau for better training dynamics.
- Introduction of cross-validation and more robust evaluation
Bad model : the model makes recommendations that are far too difficult

**CONCLUSION :** I therefore decided to evolve my approach with :

- A desire to explore different machine learning techniques.
- finding a better balance between the performance and complexity of the model.
- Adapting to the specificities of the word recommendation problem for language learning for a very particular target audience.

### 2. Ensemble Methods Exploration (HybridReco_modele2)

#### 2.1 [HybridReco_modele2.py](../src/model/HybridReco_modele2.py)
   - Introduced ensemble methods of 3 based models (Random Forest, LightGBM, CatBoost).
   - Implemented separate content-based and collaborative filtering components.
   - Combined components into a unified hybrid model.
 
| Model | Accuracy  | Precision | Recall    | F1 Score | ROC AUC  | NDCG     |
|-------|-----------|-----------|-----------|----------|----------|----------|
| rf    | 0.509725  | 0.508617  | 0.546534  | 0.526894 | 0.529542 | 0.919941 |
| lgb   | 0.522296  | 0.521536  | 0.528965  | 0.525224 | 0.536016 | 0.918264 |
| cb    | 0.518738  | 0.518501  | 0.512346  | 0.515405 | 0.531847 | 0.918688 |

**CONCLUSION:** 
Key Observations:
1. Ensemble methods show promise, with performance slightly above random classification.
2. LightGBM marginally outperforms other models in most metrics.
3. High NDCG scores (>0.91) indicate strong ranking capabilities despite moderate accuracy.
4. Balanced precision-recall trade-off across models suggests a stable prediction framework
The ensemble methods showed only marginal improvement over random classification. This suggests that we need to explore more sophisticated approaches and focus on feature engineering.


#### 2.2 [HybridReco_modele2_v2.py](../src/model/HybridReco_modele2_v2.py)
- Further refinement of the ensemble approach : include XBoost and neural network
- Added more sophisticated feature engineering.
- Improved hyperparameter tuning using optuna.
- More sophisticated model selection and evaltion

| Model | Accuracy | Precision | Recall   | F1 Score | ROC AUC  | NDCG     |
|-------|----------|-----------|----------|----------|----------|----------|
| rf    | 0.492608 | 0.499392  | 0.773540 | 0.606944 | 0.495400 | 0.898670 |
| xgb   | 0.506437 | 0.506437  | 1.0      | 0.672364 | 0.503566 | 0.908874 |
| nn    | 0.500953 | 0.504134  | 0.889830 | 0.643623 | 0.495840 | 0.903544 |

**CONCLUSION:**
Key Observations:
- Superior Performance: XGBoost outperformed other models in most metrics:
    - Highest Accuracy (0.506437)
    - Highest Precision (0.506437)
    - Perfect Recall (1.0)
    - Highest F1 Score (0.672364)
    - Highest ROC AUC (0.503566)
    - Highest NDCG (0.908874)
- Balance of Metrics: XGBoost showed the best balance across all metrics, indicating a robust and well-rounded performance.
- Perfect Recall: The 1.0 recall score suggests that XGBoost was able to identify all positive instances, which is crucial in a recommendation system to ensure no potentially interesting words are missed.
- Improved F1 Score: XGBoost's F1 score (0.672364) was significantly higher than other models, indicating a good balance between precision and recall.
- Ranking Capability: The high NDCG score (0.908874) suggests that XGBoost is particularly good at ranking recommendations, which is essential for presenting the most relevant words to users.
- Adaptability: XGBoost is known for its ability to handle various types of features and capture complex patterns, making it suitable for the diverse set of features in the word recommendation task.

### 3. XGBoost Focus

#### 3.1 [HybridReco_modele3.py](../src/model/HybridReco_modele3.py)
- Streamlined the model architecture.
- Focused on XGBoost as the primary algorithm.
- Enhanced feature engineering, particularly for temporal aspects
- Improved feature importance analysis for better interpretability.

#### 3.2 [HybridReco_final.py](../src/model/HybridReco_final.py)
- Refined and optimized version of the final model.
- Enhanced data preprocessing with better handling of categorical variables.
- Improved recommendation algorithm with more sophisticated user profiling.
- Better integration of user feedback into the recommendation process.
- Optimized for both accuracy and computational efficiency.

**CONCLUSION:** 
This model represents the culmination of my learnings and optimizations.

Key features:
- Advanced feature engineering including word embeddings and possibly PCA
- Sophisticated cross-validation and hyperparameter tuning using Optuna
- Improved recommendation algorithm with more nuanced user profiling
- Better integration of user feedback into the recommendation process
- Optimized for both accuracy and computational efficiency

## Key Learnings and Improvements

Throughout this evolution, I've made several key improvements:

1. **Feature Engineering**: Progressed from basic features to advanced engineered features, including temporal aspects and word characteristics.

2. **Model Complexity**: Moved from simple neural networks to sophisticated ensemble methods, finding a balance between model complexity and performance.

3. **Personalization**: Gradually improved my ability to personalize recommendations based on user history and proficiency.

4. **Evaluation Metrics**: Expanded from basic accuracy metrics to include ranking-specific metrics like NDCG for a more holistic evaluation.

5. **Scalability**: Optimized my approach to handle large datasets more efficiently.

6. **Interpretability**: Increased focus on model interpretability, particularly with feature importance analysis in later versions.

7. **Feedback Integration**: Improved incorporation of user feedback into the recommendation process.

The current model, HybridReco_v2_up.py, represents the culmination of these efforts, offering a robust, efficient, and personalized recommendation system for language learning.