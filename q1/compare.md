# Comparison of Tokenization Algorithms

## BPE Tokenization

- **Tokens**: ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.']
- **Token IDs**: [24, 28, 36, 34, 37, 33, 42, 32, 39, 41, 5]
- **Total Token Count**: 11

## WORDPIECE Tokenization

- **Tokens**: ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.']
- **Token IDs**: [35, 37, 41, 40, 43, 39, 53, 38, 52, 51, 5]
- **Total Token Count**: 11

## SENTENCEPIECE Tokenization

- **Tokens**: ['T', 'h', 'e', 'ca', 't', 's', 'a', 't', 'o', 'n', 't', 'h', 'e', 'm', 'a', 't', 'b', 'e', 'ca', 'u', 's', 'e', 'i', 't', 'w', 'a', 's', 't', 'i', 'r', 'e', 'd', '.']
- **Token IDs**: [22, 9, 6, 11, 5, 8, 7, 5, 16, 13, 5, 9, 6, 14, 7, 5, 17, 6, 11, 20, 8, 6, 10, 5, 12, 7, 8, 5, 10, 21, 6, 19, 15]
- **Total Token Count**: 33

## Why the Splits Differ


The tokenization algorithms split text differently due to their underlying mechanisms:

1. **BPE (Byte Pair Encoding)** works by iteratively merging the most frequent pairs of characters or character sequences. It starts with individual characters and builds up larger tokens based on frequency, making it efficient for handling rare words by breaking them into subword units.

2. **WordPiece** is similar to BPE but uses a different scoring mechanism. It chooses merges based on the likelihood of the resulting subword units rather than just frequency. This makes it better at handling morphologically rich languages.

3. **SentencePiece (Unigram)** treats tokenization as a probabilistic model, optimizing a unigram language model to find the most likely segmentation. It works directly on raw text without pre-tokenization, making it language-agnostic and capable of handling spaces as part of the tokens.

These differences lead to varying token boundaries, especially for uncommon words or morphological variants.
