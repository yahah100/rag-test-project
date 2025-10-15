"""
RAG System Benchmark Script

Evaluates the RAG system against ground truth Q&A pairs with modular scoring metrics.
Supports multiple evaluation metrics that can be easily extended.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tqdm import tqdm
from ask_rag import PDFRAG
from evaluation.metric import EvaluationMetric, ROUGEMetric, METEORMetric

# Initialize logger at module level
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Stores the result of evaluating a single Q&A pair."""
    question: str
    expected_answer: str
    generated_answer: str
    source_documents: List[str]
    page_reference: str
    category: str
    source_type: str  # 'text', 'table', or 'unknown'
    scores: Dict[str, float]  # Metric name -> score
    metadata: Dict[str, Any]  # Additional metadata


@dataclass
class BenchmarkSummary:
    """Summary statistics for the entire benchmark run."""
    total_questions: int
    successful_queries: int
    failed_queries: int
    average_scores: Dict[str, float]  # Metric name -> average score
    scores_by_category: Dict[str, Dict[str, float]]  # Category -> Metric -> score
    scores_by_source_type: Dict[str, Dict[str, float]]  # Source type -> Metric -> score
    execution_time_seconds: float
    timestamp: str


class RAGBenchmark:
    """
    Benchmark framework for evaluating RAG systems.
    
    Provides a modular architecture for adding and computing evaluation metrics.
    """
    
    def __init__(
        self,
        rag_system: PDFRAG,
        metrics: List[EvaluationMetric] | None = None
    ):
        """
        Initialize the benchmark.
        
        Args:
            rag_system: Initialized RAG system to evaluate
            metrics: List of evaluation metrics to use
        """
        self.rag_system = rag_system
        self.metrics = metrics or self._get_default_metrics()
        self.results: List[EvaluationResult] = []
        
        logger.info(f"📊 Initialized benchmark with {len(self.metrics)} metrics:")
        for metric in self.metrics:
            logger.info(f"  - {metric.name}: {metric.description}")
    
    def _get_default_metrics(self) -> List[EvaluationMetric]:
        """Get default set of evaluation metrics."""
        return [
            ROUGEMetric(rouge_type="rouge1"),
            ROUGEMetric(rouge_type="rouge2"),
            ROUGEMetric(rouge_type="rougeL"),
            METEORMetric(),
        ]
    
    
    def evaluate_single(
        self,
        question: str,
        expected_answer: str,
        page_reference: str = "",
        category: str = "general",
        source_type: str = "unknown"
    ) -> EvaluationResult:
        """
        Evaluate a single Q&A pair.
        
        Args:
            question: Question to ask the RAG system
            expected_answer: Expected/ground truth answer
            page_reference: Reference to source page
            category: Question category
            source_type: Type of source ('text', 'table', or 'unknown')
            
        Returns:
            EvaluationResult with scores for all metrics
        """
        # Query the RAG system
        try:
            result = self.rag_system.query(question)
            generated_answer = result["answer"]
            source_docs = [doc.metadata.get("filename", "Unknown") 
                          for doc in result.get("source_documents", [])]
        except Exception as e:
            logger.error(f"Failed to query: {str(e)}")
            generated_answer = f"ERROR: {str(e)}"
            source_docs = []
        
        # Compute all metric scores
        scores = {}
        metadata = {"source_documents": source_docs}
        
        for metric in self.metrics:
            try:
                score = metric.compute(expected_answer, generated_answer, metadata)
                scores[metric.name] = score
            except Exception as e:
                logger.warning(f"Metric {metric.name} failed: {str(e)}")
                scores[metric.name] = 0.0
        
        # Create evaluation result
        eval_result = EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            source_documents=source_docs,
            page_reference=page_reference,
            category=category,
            source_type=source_type,
            scores=scores,
            metadata=metadata
        )
        
        self.results.append(eval_result)
        return eval_result
    
    def run_benchmark(
        self,
        qa_pairs: List[Dict[str, Any]],
    ) -> BenchmarkSummary:
        """
        Run benchmark on a set of Q&A pairs.
        
        Args:
            qa_pairs: List of Q&A pairs from ground truth
            max_questions: Maximum number of questions to evaluate (None = all)
            sample_strategy: How to sample questions if max_questions is set
            
        Returns:
            BenchmarkSummary with aggregated results
        """
        start_time = time.time()
        logger.info(f"🚀 Running benchmark on {len(qa_pairs)} questions...")
        
        successful = 0
        failed = 0
        
        # Evaluate each Q&A pair
        for qa in tqdm(qa_pairs, desc="Evaluating questions"):
            try:
                self.evaluate_single(
                    question=qa["question"],
                    expected_answer=qa["answer"],
                    page_reference=qa.get("page_reference", ""),
                    category=qa.get("category", "general"),
                    source_type=qa.get("source_type", "unknown")
                )
                successful += 1
            except Exception as e:
                logger.error(f"Failed to evaluate question: {str(e)}")
                failed += 1
        
        end_time = time.time()
        
        # Compute summary statistics
        summary = self._compute_summary(
            total=len(qa_pairs),
            successful=successful,
            failed=failed,
            execution_time=end_time - start_time
        )
        
        return summary
    
    def _compute_summary(
        self,
        total: int,
        successful: int,
        failed: int,
        execution_time: float
    ) -> BenchmarkSummary:
        """Compute summary statistics from results."""
        
        # Compute average scores across all metrics
        avg_scores = {}
        for metric in self.metrics:
            scores = [r.scores[metric.name] for r in self.results if metric.name in r.scores]
            avg_scores[metric.name] = sum(scores) / len(scores) if scores else 0.0
        
        # Compute scores by category
        category_scores = {}
        categories = set(r.category for r in self.results)
        
        for category in categories:
            category_results = [r for r in self.results if r.category == category]
            category_scores[category] = {}
            
            for metric in self.metrics:
                scores = [r.scores[metric.name] for r in category_results 
                         if metric.name in r.scores]
                category_scores[category][metric.name] = sum(scores) / len(scores) if scores else 0.0
        
        # Compute scores by source type
        source_type_scores = {}
        source_types = set(r.source_type for r in self.results)
        
        for source_type in source_types:
            source_type_results = [r for r in self.results if r.source_type == source_type]
            source_type_scores[source_type] = {}
            
            for metric in self.metrics:
                scores = [r.scores[metric.name] for r in source_type_results 
                         if metric.name in r.scores]
                source_type_scores[source_type][metric.name] = sum(scores) / len(scores) if scores else 0.0
        
        return BenchmarkSummary(
            total_questions=total,
            successful_queries=successful,
            failed_queries=failed,
            average_scores=avg_scores,
            scores_by_category=category_scores,
            scores_by_source_type=source_type_scores,
            execution_time_seconds=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def save_results(self, output_file: str):
        """Save detailed results to JSON file."""
        results_dict = {
            "results": [asdict(r) for r in self.results],
            "metrics": [{"name": m.name, "description": m.description} for m in self.metrics]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Saved detailed results to {output_file}")
    
    def save_summary(self, summary: BenchmarkSummary, output_file: str):
        """Save summary statistics to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        logger.info(f"💾 Saved summary to {output_file}")
    
    def print_summary(self, summary: BenchmarkSummary):
        """Print formatted summary to console."""
        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"\n⏱️  Execution time: {summary.execution_time_seconds:.2f} seconds")
        print(f"📝 Total questions: {summary.total_questions}")
        print(f"✅ Successful: {summary.successful_queries}")
        print(f"❌ Failed: {summary.failed_queries}")
        
        print("\n📈 Average Scores Across All Questions:")
        print("-" * 80)
        for metric_name, score in summary.average_scores.items():
            print(f"  {metric_name:30s}: {score:.3f}")
        
        print("\n📊 Scores by Source Type:")
        print("-" * 80)
        for source_type, scores in summary.scores_by_source_type.items():
            # Count questions of this source type
            count = sum(1 for r in self.results if r.source_type == source_type)
            print(f"\n  Source Type: {source_type} ({count} questions)")
            for metric_name, score in scores.items():
                print(f"    {metric_name:28s}: {score:.3f}")
        
        print("\n📊 Scores by Category:")
        print("-" * 80)
        for category, scores in summary.scores_by_category.items():
            print(f"\n  Category: {category}")
            for metric_name, score in scores.items():
                print(f"    {metric_name:28s}: {score:.3f}")
        
        print("\n" + "=" * 80)


def load_ground_truth(json_file: str) -> List[Dict[str, Any]]:
    """
    Load ground truth Q&A pairs from JSON file.
    
    Args:
        json_file: Path to ground truth JSON file
        
    Returns:
        List of Q&A pairs with metadata
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    qa_pairs = []
    for doc in data.get("documents", []):
        for qa in doc.get("qa_pairs", []):
            qa_pairs.append(qa)
    
    logger.info(f"📚 Loaded {len(qa_pairs)} Q&A pairs from {json_file}")
    return qa_pairs


def main():
    """Main function for running the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark RAG System")
    parser.add_argument("--ground-truth", type=str, default="datasets/qa_ground_truth.json",
                       help="Path to ground truth JSON file")
    parser.add_argument("--pdf-folder", type=str, default="datasets/PMOC_samples",
                       help="Path to PDF folder")
    parser.add_argument("--ollama-model", type=str, default="gemma3:4b",
                       help="Ollama model for generation")
    parser.add_argument("--embedding-model", type=str, default="embeddinggemma:300m",
                       help="Embedding model")
    parser.add_argument("--persist-directory", type=str, default="./chroma_db",
                       help="Vector store directory")
    parser.add_argument("--k-similar-chunks", type=int, default=2,
                       help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=4000,
                       help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=500,
                       help="Chunk overlap in characters")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild vector store")
    
    args = parser.parse_args()
    
    print("🧪 RAG System Benchmark")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load ground truth
    logger.info("📚 Loading ground truth data...")
    qa_pairs = load_ground_truth(args.ground_truth)
    
    # Initialize RAG system
    logger.info("🚀 Initializing RAG system...")
    rag_system = PDFRAG(
        pdf_folder=args.pdf_folder,
        ollama_model=args.ollama_model,
        embedding_model=args.embedding_model,
        persist_directory=args.persist_directory,
        k_similar_chunks=args.k_similar_chunks,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    rag_system.initialize_system(force_rebuild=args.force_rebuild)
    
    # Create benchmark
    benchmark = RAGBenchmark(rag_system)
    
    # Run benchmark
    summary = benchmark.run_benchmark(
        qa_pairs=qa_pairs,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"benchmark_results_{timestamp}.json"
    summary_file = output_dir / f"benchmark_summary_{timestamp}.json"
    
    benchmark.save_results(str(results_file))
    benchmark.save_summary(summary, str(summary_file))
    
    # Print summary
    benchmark.print_summary(summary)
    
    print(f"\n💾 Results saved to:")
    print(f"  - {results_file}")
    print(f"  - {summary_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    main()
