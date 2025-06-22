# Tokenization & Fill-in-the-Blank Assignment

This project demonstrates different tokenization algorithms (BPE, WordPiece, and SentencePiece) and uses a language model to predict masked tokens in a sentence.

## Overview

The project explores how different tokenization methods process text and how language models can predict missing words in a sentence. It includes:

1. Implementation of three tokenization algorithms
2. Comparison of their outputs on a sample sentence
3. Demonstration of fill-in-the-blank prediction using a language model

## Files

- `tokenise.py`: Python script that implements tokenization and prediction
- `compare.md`: Detailed comparison of tokenization algorithm outputs
- `predictions.json`: Results of the fill-mask prediction task
- `requirements.txt`: Required Python packages

## How to Run

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the tokenization script:
   ```
   python tokenise.py
   ```

## Implementation Details

### Tokenization Algorithms

The project implements three tokenization algorithms:

1. **BPE (Byte Pair Encoding)**:
   - Iteratively merges the most frequent pairs of characters
   - Builds vocabulary from character level up based on frequency
   - Efficient for handling rare words by breaking them into subword units

2. **WordPiece**:
   - Similar to BPE but uses likelihood-based scoring
   - Chooses merges based on the likelihood of resulting subword units
   - Better at handling morphologically rich languages

3. **SentencePiece (Unigram)**:
   - Probabilistic model that optimizes a unigram language model
   - Works directly on raw text without pre-tokenization
   - Language-agnostic and handles spaces as part of tokens

### Fill-in-the-Blank Prediction

The script demonstrates how to:
1. Create masked versions of a sentence by replacing words with [MASK] tokens
2. Use a language model to predict the most likely words for each mask
3. Display the top predictions with their probability scores

## Findings

### Tokenization Results

For the sentence "The cat sat on the mat because it was tired.":

- **BPE and WordPiece** both produced 11 tokens matching the words in the sentence
- **SentencePiece** produced different tokenization with character-level splits
- The token IDs differ between methods as they build vocabularies independently

### Prediction Results

For masked words in the sentence:
- The model successfully predicts contextually appropriate words
- Top predictions for masked words are semantically relevant
- Prediction confidence varies based on contextual constraints

## Learning Outcomes

- Understanding different tokenization algorithms and their trade-offs
- Seeing how language models represent text as tokens
- Learning how models predict missing words in a sentence
- Gaining practical experience with NLP libraries

## Future Work

- Train tokenizers on larger datasets to see more distinctive differences
- Experiment with different masking strategies
- Compare predictions from different language models
- Analyze how tokenization affects downstream tasks
