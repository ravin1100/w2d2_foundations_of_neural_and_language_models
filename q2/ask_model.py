import json
import os
import time
import random
from validator import Validator

# Simulating a language model API
# In a real implementation, you would use an actual API like OpenAI's
class MockLanguageModel:
    def __init__(self, kb_path="kb.json", accuracy=0.7):
        """
        A mock language model that simulates responses.
        
        Args:
            kb_path: Path to knowledge base
            accuracy: Probability of giving correct answer for in-KB questions
        """
        self.accuracy = accuracy
        
        # Load the knowledge base for simulation
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)
        
        self.kb_dict = {item["question"]: item["answer"] for item in self.kb}
    
    def ask(self, question):
        """
        Simulate asking the model a question.
        
        Args:
            question: The question to ask
            
        Returns:
            str: Simulated model response
        """
        # If question is in KB, return correct answer with probability=accuracy
        if question in self.kb_dict:
            if random.random() < self.accuracy:
                return self.kb_dict[question]
            else:
                # Return a slightly incorrect answer
                return f"{self.kb_dict[question]} (with some errors)"
        else:
            # For out-of-domain questions, generate a made-up answer
            return f"I think the answer is [hallucinated response for {question}]"


def main():
    # Initialize the validator and mock model
    validator = Validator()
    model = MockLanguageModel()
    
    # Load questions from knowledge base
    with open("kb.json", 'r') as f:
        kb_data = json.load(f)
    
    # Extract questions from KB
    kb_questions = [item["question"] for item in kb_data]
    
    # Add 5 out-of-domain questions
    additional_questions = [
        "What is the population of Mars?",
        "Who is the current CEO of Fictional Corp?",
        "What is the recipe for a perfect soufflé?",
        "When will humans achieve immortality?",
        "What is the meaning of life?"
    ]
    
    all_questions = kb_questions + additional_questions
    
    # Open log file
    with open("run.log", 'w') as log:
        log.write("=== Model Query Log ===\n\n")
        
        # Process each question
        for question in all_questions:
            log.write(f"Question: {question}\n")
            
            # First attempt
            answer = model.ask(question)
            log.write(f"First answer: {answer}\n")
            
            # Validate the answer
            is_valid, message = validator.validate(question, answer)
            
            if is_valid:
                log.write("Validation: PASS\n")
            else:
                log.write(f"Validation: {message}\n")
                
                # Retry once if validation failed
                log.write("Retrying...\n")
                retry_answer = model.ask(question)
                log.write(f"Second answer: {retry_answer}\n")
                
                # Validate retry
                retry_valid, retry_message = validator.validate(question, retry_answer)
                if retry_valid:
                    log.write("Retry validation: PASS\n")
                else:
                    log.write(f"Retry validation: {retry_message}\n")
            
            log.write("\n---\n\n")
    
    print("Model queries complete. Results saved to run.log")
    
    # Generate summary statistics
    generate_summary()


def generate_summary():
    """Generate summary statistics and write to summary.md"""
    # Parse the log file to extract statistics
    with open("run.log", 'r') as log:
        content = log.read()
    
    # Count various outcomes
    kb_questions_count = 10
    total_questions = 15
    out_of_domain_count = 5
    
    first_pass_count = content.count("Validation: PASS")
    retry_pass_count = content.count("Retry validation: PASS")
    differs_from_kb_count = content.count("RETRY: answer differs from KB")
    
    # Write summary
    with open("summary.md", 'w') as summary:
        summary.write("# Hallucination Detection Summary\n\n")
        summary.write("## Statistics\n\n")
        summary.write(f"- Total questions: {total_questions}\n")
        summary.write(f"- In-KB questions: {kb_questions_count}\n")
        summary.write(f"- Out-of-domain questions: {out_of_domain_count}\n")
        summary.write(f"- First attempt correct: {first_pass_count}\n")
        summary.write(f"- Correct after retry: {retry_pass_count}\n")
        summary.write(f"- Answers differing from KB: {differs_from_kb_count}\n\n")
        
        summary.write("## Analysis\n\n")
        summary.write("The validation system uses simple string matching to detect hallucinations. ")
        summary.write("It checks if the model's answer matches what's in our knowledge base and flags ")
        summary.write("any discrepancies as potential hallucinations.\n\n")
        
        summary.write("### Strengths\n\n")
        summary.write("- Simple to implement\n")
        summary.write("- Clear distinction between in-KB and out-of-domain questions\n")
        summary.write("- Retry mechanism helps catch and correct hallucinations\n\n")
        
        summary.write("### Limitations\n\n")
        summary.write("- String matching is brittle and may flag correct answers with different wording\n")
        summary.write("- Cannot validate the accuracy of out-of-domain answers\n")
        summary.write("- No semantic understanding of answers\n\n")
        
        summary.write("### Improvements\n\n")
        summary.write("- Implement semantic matching instead of string matching\n")
        summary.write("- Add confidence scores to model responses\n")
        summary.write("- Expand the knowledge base with reliable sources\n")
        summary.write("- Implement more sophisticated retry strategies\n")


if __name__ == "__main__":
    main() 