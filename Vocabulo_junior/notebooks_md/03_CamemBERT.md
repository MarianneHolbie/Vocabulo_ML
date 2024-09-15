# CamemBERT Implementation in Vocabulo Junior

## Overview
CamemBERT is a state-of-the-art French language model. In the final application, it plays a crucial role in generating 
contextual word embeddings and improving Word Sense Disambiguation (WSD) for French text, enhancing the app's ability 
to match words to corresponding LSF (Langue des Signes Française) signs.

## Workflow

### Process Flow Diagram

````mermaid
graph TD
    A[OCR-extracted text files] --> B[preprocess_data_Bert.py]
    B -->|Clean and split text| C[prepared_data_for_camembert_test.txt]
    C --> D[CamemBert_model.py]
    D -->|Generate embeddings| E[camembert_processed_data_test.txt]
    E --> F[Get_LSFsign.py]
    G[(PostgreSQL Database)] --> F
    F -->|Analyze and match| H[CSV with LSF signs]

    subgraph "CamemBERT Processing"
    D
    end

    subgraph "Word Sense Disambiguation"
    F
    end
````
### Data Preprocessing ([preprocess_data_Bert.py](../src/nlp/camemBERT/preprocess_data_Bert.py))

- Input: Raw text extracted through OCR from children’s books.
- Process: Text cleaning and normalization, removal of special characters, and sentence segmentation to prepare for model input.
- Output: A cleaned, sentence-split file, `prepared_data_for_camembert_test.txt`, optimized for embedding generation.

### CamemBERT Processing ([CamemBert_model.py](../src/nlp/camemBERT/CamemBert_model.py))

- Input: Preprocessed text file (prepared_data_for_camembert_test.txt).
- Process:
  - Sentences are tokenized and fed into the CamemBERT model.
  - The model generates contextual embeddings for each word based on its surrounding words, capturing nuances in meaning.
- Output: Embeddings of words along with their sentence context in `camembert_processed_data_test.txt`.


## Key Components

  ### CamemBERT Model

  - Pretraining: CamemBERT is pretrained on a large French corpus, including books, news, and web pages, allowing it to
generate rich, contextual embeddings for each word.
  - Contextual Embeddings: The model creates representations of words based on the surrounding text, allowing it to 
differentiate between different meanings of a word in various contexts.
  - Word Representation: These embeddings are crucial for tasks like WSD, where context plays a vital role in 
understanding a word’s intended meaning.

  ### Word Sense Disambiguation (WSD)

  - Technique: CamemBERT generates contextual embeddings for each word, which are then compared with the predefined 
meanings (from the database) using cosine similarity.
  - Process: The system selects the word definition or meaning that has the highest similarity to the context where 
the word appears, ensuring the most accurate match to the LSF sign.
  - Challenges: Handling polysemy (words with multiple meanings) in children’s literature, which often includes creative
language and unique expressions.

 ### Benefits of Using CamemBERT

  1. **Improved WSD Accuracy**: The contextual nature of CamemBERT’s embeddings significantly enhances the ability to
disambiguate between word meanings, crucial for matching words with their correct LSF signs.
  2. **Handling French Language Nuances**: French’s complex grammar, including gender, conjugations, and polysemy, is 
better captured through the pretraining of CamemBERT on vast French corpora.
  3. **Context-Aware Embeddings**: The model doesn't treat words in isolation but instead generates embeddings that 
consider the entire sentence, leading to more precise word representations.
  4. **Enhanced LSF Sign Matching**: With more accurate word sense identification, the app can match words to their 
appropriate LSF signs, providing a more tailored learning experience for users.
  
### Future Improvements
  - **Domain-Specific Fine-Tuning**: Fine-tuning CamemBERT on a dataset specific to children’s literature or sign 
language corpora could further enhance performance by better adapting to the nuances of the domain.
  - **Advanced Disambiguation Techniques**: Experimenting with other models or hybrid approaches (combining rule-based 
and machine-learning techniques) to improve the accuracy and speed of word-sense disambiguation.
  - **Processing Optimization**: As the dataset grows, optimizing the model’s processing speed and memory usage will 
ensure scalability, allowing Vocabulo Junior to handle more text in real-time.