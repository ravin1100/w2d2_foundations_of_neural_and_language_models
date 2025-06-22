import json
import re

class Validator:
    def __init__(self, kb_path="kb.json"):
        """
        Initialize the validator with a knowledge base.
        
        Args:
            kb_path: Path to the knowledge base JSON file
        """
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)
        
        # Create a dictionary for faster lookups
        self.kb_dict = {item["question"]: item["answer"] for item in self.kb}
    
    def normalize_text(self, text):
        """
        Normalize text for better matching by removing punctuation,
        extra whitespace, and converting to lowercase.
        """
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with single space
        return text.strip().lower()
    
    def validate(self, question, answer):
        """
        Validate if the answer matches the knowledge base.
        
        Args:
            question: The question asked
            answer: The model's answer
            
        Returns:
            tuple: (is_valid, message)
                is_valid: Boolean indicating if answer is valid
                message: Validation message or None
        """
        # Check if question is in knowledge base
        if question in self.kb_dict:
            kb_answer = self.kb_dict[question]
            
            # Normalize both answers for comparison
            norm_kb_answer = self.normalize_text(kb_answer)
            norm_model_answer = self.normalize_text(answer)
            
            # Check if the normalized answers match
            if norm_kb_answer in norm_model_answer or norm_model_answer in norm_kb_answer:
                return True, None
            else:
                return False, "RETRY: answer differs from KB"
        else:
            return False, "RETRY: out-of-domain" 