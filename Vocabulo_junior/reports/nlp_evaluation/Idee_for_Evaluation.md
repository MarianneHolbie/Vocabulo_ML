# Evaluation Strategy for NLP Components in Vocabulo Junior

## 1. Creating a Test Dataset
Build a corpus of children's books in French, representative of target domain.
Manually annotate this corpus with the correct linguistic information (POS, lemmas, word senses, corresponding LSF signs).

## 2. spaCy Evaluation
Metrics:
    - Precision, recall, and F1-score for:
    - Tokenization
    - POS tagging
    - Lemmatization
    - Dependency parsing

Method:
```python

from spacy.scorer import Scorer
from spacy.tokens import Doc

scorer = Scorer()
with test_corpus as test_data:
    for text, annot in test_data:
        doc_gold = Doc.from_dict(spacy_model.vocab, annot)
        doc_pred = spacy_model(text)
        scores = scorer.score(doc_pred, doc_gold)
print(scores)
```

## 3. CamemBERT Evaluation
Metrics:
- Perplexity on the test corpus
- Accuracy for the lexical disambiguation task

Method:
```python

from transformers import CamembertForMaskedLM, CamembertTokenizer
import torch

model = CamembertForMaskedLM.from_pretrained("camembert-base")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss)

perplexities = [calculate_perplexity(text) for text in test_corpus]
average_perplexity = sum(perplexities) / len(perplexities)
print(f"Average Perplexity: {average_perplexity}")
```

## 4. Lexical Disambiguation Evaluation
Metrics:
- Lexical disambiguation accuracy
- Top-k accuracy (k=1, 3, 5)

Method:
```python

def evaluate_wsd(predictions, gold_standard):
    correct = sum(1 for pred, gold in zip(predictions, gold_standard) if pred == gold)
    return correct / len(predictions)

wsd_accuracy = evaluate_wsd(wsd_predictions, gold_standard_senses)
print(f"Lexical Disambiguation Accuracy: {wsd_accuracy}")
```

## 5. LSF Matching Evaluation
Metrics:
- Accuracy of word-sign LSF matching

## 6. End-to-End Evaluation
Metrics:
- User satisfaction rate
- Average processing time
- Overall error rate

Method:
- Set up user testing with children and educators.
- Collect their feedback through surveys and direct observation.
- Measure processing time for different text lengths.
- Calculate the overall error rate by comparing the final results with a manually annotated LSF reference.

## 7. Error Analysis
-Identify the most frequent error types for each component.
-Create a confusion matrix to understand classification errors.
-Conduct a qualitative analysis of problematic cases to identify areas for improvement.

## 8. Continuous Evaluation
- Set up logging to collect statistics in production.
- Perform periodic evaluations on new datasets.
- Use user feedback to continuously enrich test corpus.

**CONCLUSION:**
This evaluation strategy will enable me to accurately measure the performance of each NLP component and of the system as a whole.
It will also enable me to identify areas for improvement and monitor progress over time. 
This strategy has not yet been implemented.