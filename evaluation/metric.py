import logging
from typing import Any, Dict

import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class EvaluationMetric:
    """Base class for evaluation metrics."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def compute(self, expected: str, generated: str, metadata: Dict[str, Any] | None = None) -> float:
        """
        Compute the metric score.
        
        Args:
            expected: Expected/ground truth answer
            generated: Generated answer from RAG system
            metadata: Additional metadata (e.g., source documents)
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        raise NotImplementedError("Subclasses must implement compute()")



class ROUGEMetric(EvaluationMetric):
    """Computes ROUGE scores for evaluating answer quality."""
    
    def __init__(self, rouge_type: str = "rougeL"):
        """
        Initialize ROUGE metric.
        
        Args:
            rouge_type: Type of ROUGE score ('rouge1', 'rouge2', 'rougeL')
        """
        super().__init__(
            name=f"rouge_{rouge_type}",
            description=f"ROUGE-{rouge_type.upper()} F1 score"
        )
        self.rouge_type = rouge_type
        
        # Import rouge_score here to make it optional
        try:
            self.scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
        except ImportError:
            logging.warning("rouge_score not installed. Install with: pip install rouge-score")
            self.scorer = None
    
    def compute(self, expected: str, generated: str, metadata: Dict[str, Any] | None = None) -> float:
        if self.scorer is None:
            return 0.0
        
        try:
            scores = self.scorer.score(expected, generated)
            # Return F1 score for the specified ROUGE type
            return scores[self.rouge_type].fmeasure
        except Exception as e:
            logging.warning(f"ROUGE computation failed: {str(e)}")
            return 0.0


class METEORMetric(EvaluationMetric):
    """Computes METEOR score for evaluating answer quality."""
    
    def __init__(self):
        super().__init__(
            name="meteor",
            description="METEOR score with stemming and synonym matching"
        )
        
        # Import nltk here to make it optional
        try:
            try:
                nltk.data.find('wordnet')
            except LookupError:
                logging.info("Downloading NLTK wordnet data...")
                nltk.download('wordnet', quiet=True)
            
            try:
                nltk.data.find('omw-1.4')
            except LookupError:
                logging.info("Downloading NLTK omw-1.4 data...")
                nltk.download('omw-1.4', quiet=True)
            
            self.meteor_score_func = meteor_score
            self.available = True
        except ImportError:
            logging.warning("NLTK not installed. Install with: pip install nltk")
            self.available = False
    
    def compute(self, expected: str, generated: str, metadata: Dict[str, Any] | None = None) -> float:
        if not self.available:
            return 0.0
        
        try:
            # METEOR expects lists of tokens
            expected_tokens = expected.split()
            generated_tokens = generated.split()
            
            # meteor_score expects reference as list of tokens, hypothesis as list of tokens
            score = self.meteor_score_func([expected_tokens], generated_tokens)
            return score
        except Exception as e:
            logging.warning(f"METEOR computation failed: {str(e)}")
            return 0.0