# Hallucination Detection System

This project implements a simple system to detect hallucinations in language model responses by verifying answers against a knowledge base.

## Project Structure

- `kb.json`: Knowledge base containing 10 factual question-answer pairs
- `validator.py`: Implements validation logic using string matching
- `ask_model.py`: Script to query a language model with questions and validate responses
- `run.log`: Log file containing model queries and validation results
- `summary.md`: Summary of findings and analysis

## How It Works

1. The system loads questions from a knowledge base (`kb.json`)
2. It asks these questions plus 5 out-of-domain questions to a language model
3. For each response, it validates if the answer matches what's in the knowledge base
4. If validation fails, it retries once
5. All interactions are logged to `run.log`
6. Summary statistics are generated in `summary.md`

## Validation Logic

- If a question is in the knowledge base but the answer doesn't match → "RETRY: answer differs from KB"
- If a question is not in the knowledge base → "RETRY: out-of-domain"

## Running the System

```bash
python ask_model.py
```

## Extending the System

To improve the system:
1. Replace the mock language model with a real API (e.g., OpenAI)
2. Implement more sophisticated matching algorithms
3. Expand the knowledge base
4. Add confidence scores to model responses 