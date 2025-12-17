import os
import time
from vertexai.generative_models import GenerativeModel

class LLMJudge:
    def __init__(self):
        self.system_prompt = (
            "Instruction: You will receive a question, a ground truth answer to that question, and a candidate answer to that same question. Your task is to determine if the candidate answer matches the meaning of the ground truth answer. Return 'True' if they match, or 'False' if they do not. Provide no other text."
        )
        self.model = GenerativeModel(
            model_name="gemini-1.5-flash-002",
            # model_name="gemini-2.0-flash-exp",
            system_instruction=self.system_prompt
        )
    
    def process_response(self, response):
        response = response.strip().lower()
        if response == "true":
            return True
        elif response == "false":
            return False 
        else:
            raise ValueError("Invalid response: {}".format(response))
        
    def compare_responses(self, question: str, ground_truth: str, candidate_answer: str):
        try:
            prompt_parts = [
                f"Question: {question}",
                f"Ground truth: {ground_truth}",
                f"Candidate answer: {candidate_answer}"
            ]
            response = self.model.generate_content(prompt_parts)
            return self.process_response(response.text)
        except Exception as e:
            raise RuntimeError(str(e))
        
    def process_batch(self, batch_data: list):
        """
        Processes a batch of questions, ground truths, and candidate answers.

        Args:
            batch_data (list of dict): A list where each dictionary contains:
                - "question" (str): The question being asked.
                - "ground_truth" (str): The correct answer to the question.
                - "candidate_answer" (str): The answer to be evaluated.

        Returns:
            list of bool: A list of boolean values indicating whether each candidate answer matches the ground truth.
        """
        results = []
        for data in batch_data:
            result = self.compare_responses(data["question"], data["ground_truth"], data["candidate_answer"])
            results.append(result)
            time.sleep(0.4)
        return results