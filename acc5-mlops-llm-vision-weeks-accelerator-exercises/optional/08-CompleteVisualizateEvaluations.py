import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import argparse

# Constants
OUTPUT_DIR_TEMPLATE = "evaluation_results/visualization_{timestamp}"
LOG_FILE_NAME = "evaluation.log"
METRICS_CSV_NAME = "metrics_results.csv"
SUMMARY_REPORT_NAME = "analysis_summary.txt"
DATE_FORMAT = "%Y%m%d_%H%M%S"

# Default Evaluation File (Can be overridden via command-line arguments)
DEFAULT_EVALUATION_FILE_TEMPLATE = "evaluation_results/strategies_evaluation_20250117_235644.json"


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure logging with both file and console handlers.

    Args:
        output_dir (Path): Directory where the log file will be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('rag_evaluation')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate logs

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(output_dir / LOG_FILE_NAME)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_evaluation_results(file_path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load the evaluation results from a JSON file.

    Args:
        file_path (Path): Path to the JSON file containing evaluation results.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        List[Dict[str, Any]]: List of evaluation results for all questions.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    logger.info(f"Loading evaluation results from {file_path}")
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Assuming data contains a 'results' key with list of question results
            results = data.get('results', [])
            logger.info(f"Successfully loaded evaluation results for {len(results)} questions")
            return results
        elif isinstance(data, list):
            logger.info(f"Successfully loaded evaluation results for {len(data)} questions")
            return data
        else:
            logger.error("Unexpected JSON structure. Expected a list or a dict with 'results' key.")
            raise ValueError("Invalid JSON structure.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading evaluation results: {e}")
        raise


def create_metrics_dataframe(all_results: List[Dict[str, Any]], logger: logging.Logger) -> pd.DataFrame:
    """
    Create a DataFrame with all metrics for each strategy across all questions.

    Args:
        all_results (List[Dict[str, Any]]): List of evaluation results for all questions.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: DataFrame containing metrics data for all questions.
    """
    logger.info("Creating metrics DataFrame for all questions")
    data = []
    for question_result in all_results:
        question_index = question_result.get('question_index', 'N/A')
        query = question_result.get('query', 'N/A')
        strategies = question_result.get('strategies', {})
        for strategy, strategy_details in strategies.items():
            metrics = strategy_details.get('metrics', {})
            for metric, details in metrics.items():
                data.append({
                    'question_index': question_index,
                    'query': query,
                    'strategy': strategy,
                    'metric': metric,
                    'score': details.get('score', np.nan),
                    'reason': details.get('reason', '')
                })
    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with {len(df)} rows")
    return df


def save_metrics_csv(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Save metrics DataFrame to CSV.

    Args:
        df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the CSV will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    csv_path = output_dir / METRICS_CSV_NAME
    logger.info(f"Saving metrics to CSV: {csv_path}")
    try:
        df.to_csv(csv_path, index=False)
        logger.info("Metrics CSV saved successfully")
    except Exception as e:
        logger.error(f"Failed to save metrics CSV: {e}")
        raise


def plot_metrics_heatmap(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create an interactive heatmap using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the heatmap will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Creating metrics heatmap")

    # Calculate average score for each strategy and metric
    pivot_data = df.groupby(['strategy', 'metric'])['score'].mean().unstack()

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            text=np.round(pivot_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='RdYlGn',
            zmid=0.5,
            colorbar=dict(title='Average Score')
        )
    )

    fig.update_layout(
        title='Average RAG Strategy Performance Across Metrics',
        height=600,
        width=1000,
        xaxis_title='Metrics',
        yaxis_title='Strategy'
    )

    try:
        fig.write_html(output_dir / 'metrics_heatmap.html')
        fig.write_image(output_dir / 'metrics_heatmap.png')
        logger.info("Saved heatmap visualizations")
    except Exception as e:
        logger.error(f"Failed to save heatmap visualizations: {e}")
        raise


def plot_strategy_performance(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create an interactive bar plot showing average performance by strategy.

    Args:
        df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the plot will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Creating strategy performance plot")

    # Calculate average and standard deviation of scores for each strategy
    strategy_stats = df.groupby('strategy')['score'].agg(['mean', 'std']).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Average Score',
        x=strategy_stats['strategy'],
        y=strategy_stats['mean'],
        error_y=dict(
            type='data',
            array=strategy_stats['std'],
            visible=True
        ),
        text=strategy_stats['mean'].round(2),
        textposition='auto',
        marker_color='indianred'
    ))

    fig.update_layout(
        title='Average Performance by Strategy',
        xaxis_title='Strategy',
        yaxis_title='Average Score',
        yaxis=dict(range=[0, 1]),
        height=600,
        width=1000,
        showlegend=False
    )

    try:
        fig.write_html(output_dir / 'strategy_performance.html')
        fig.write_image(output_dir / 'strategy_performance.png')
        logger.info("Saved strategy performance visualizations")
    except Exception as e:
        logger.error(f"Failed to save strategy performance visualizations: {e}")
        raise


def plot_metric_comparison(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create an interactive grouped bar plot comparing metrics across strategies.

    Args:
        df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the plot will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Creating metric comparison plot")

    # Calculate average score for each strategy and metric
    metric_stats = df.groupby(['strategy', 'metric'])['score'].mean().reset_index()

    fig = px.bar(
        metric_stats,
        x='strategy',
        y='score',
        color='metric',
        barmode='group',
        title='Average Metric Scores by Strategy',
        labels={'strategy': 'Strategy', 'score': 'Average Score', 'metric': 'Metric'},
        height=600,
        width=1000
    )

    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        xaxis_tickangle=-45,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    try:
        fig.write_html(output_dir / 'metric_comparison.html')
        fig.write_image(output_dir / 'metric_comparison.png')
        logger.info("Saved metric comparison visualizations")
    except Exception as e:
        logger.error(f"Failed to save metric comparison visualizations: {e}")
        raise


def plot_latency_analysis(all_results: List[Dict[str, Any]], output_dir: Path, logger: logging.Logger) -> None:
    """
    Create interactive scatter plots analyzing latency vs performance metrics.

    Args:
        all_results (List[Dict[str, Any]]): List of evaluation results for all questions.
        output_dir (Path): Directory where the plots will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Creating latency analysis plots")

    # Aggregate latency and average score per strategy across all questions
    data = []
    for question_result in all_results:
        strategies = question_result.get('strategies', {})
        for strategy, strategy_details in strategies.items():
            latency = strategy_details['execution_info'].get('latency', np.nan)
            average_score = np.mean([
                metric['score'] for metric in strategy_details.get('metrics', {}).values()
                if isinstance(metric.get('score', None), (int, float))
            ]) if strategy_details.get('metrics', {}) else np.nan
            data.append({
                'strategy': strategy,
                'latency': latency,
                'average_score': average_score
            })

    latency_df = pd.DataFrame(data)
    logger.info(f"Latency analysis DataFrame created with {len(latency_df)} rows")

    # Calculate mean latency and performance per strategy
    strategy_latency = latency_df.groupby('strategy').agg({
        'latency': 'mean',
        'average_score': 'mean'
    }).reset_index()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=strategy_latency['latency'],
            y=strategy_latency['average_score'],
            mode='markers+text',
            text=strategy_latency['strategy'],
            textposition="top center",
            marker=dict(size=12, color='blue', opacity=0.7),
            name='Strategy Performance'
        )
    )

    fig.update_layout(
        title='Latency vs. Average Performance by Strategy',
        xaxis_title="Average Latency (seconds)",
        yaxis_title="Average Score",
        height=600,
        width=800,
        showlegend=False
    )

    try:
        fig.write_html(output_dir / 'performance_latency_analysis.html')
        fig.write_image(output_dir / 'performance_latency_analysis.png')
        logger.info("Saved latency analysis visualizations")
    except Exception as e:
        logger.error(f"Failed to save latency analysis visualizations: {e}")
        raise


def generate_summary_report(
    all_results: List[Dict[str, Any]],
    metrics_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Generate a detailed text summary of the aggregated analysis.

    Args:
        all_results (List[Dict[str, Any]]): List of evaluation results for all questions.
        metrics_df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the summary report will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Generating summary report")

    summary_lines = [
        "RAG Strategies Evaluation Summary",
        "=" * 40 + "\n"
    ]

    total_questions = len(all_results)
    summary_lines.append(f"Total Questions Evaluated: {total_questions}\n")
    logger.info(f"Total Questions Evaluated: {total_questions}")

    summary_lines.append("Overall Performance by Strategy")
    summary_lines.append("-" * 30)
    summary_lines.append("")

    # Calculate overall average scores and latencies per strategy
    overall_stats = metrics_df.groupby('strategy')['score'].mean().reset_index()
    latency_data = []
    for question_result in all_results:
        strategies = question_result.get('strategies', {})
        for strategy, details in strategies.items():
            latency = details['execution_info'].get('latency', np.nan)
            latency_data.append({
                'strategy': strategy,
                'latency': latency
            })
    latency_df = pd.DataFrame(latency_data)
    latency_stats = latency_df.groupby('strategy')['latency'].mean().reset_index()

    overall_summary = pd.merge(overall_stats, latency_stats, on='strategy', how='left')

    for _, row in overall_summary.iterrows():
        strategy = row['strategy']
        avg_score = row['score']
        avg_latency = row['latency']
        summary_str = (
            f"{strategy}:\n"
            f"  Average Score: {avg_score:.3f}\n"
            f"  Average Latency: {avg_latency:.3f} seconds\n"
        )
        summary_lines.append(summary_str)
        logger.info(
            f"Strategy: {strategy} | Average Score: {avg_score:.3f} | "
            f"Average Latency: {avg_latency:.3f}s"
        )

    # Identify best performing strategy
    best_strategy_row = overall_stats.loc[overall_stats['score'].idxmax()]
    best_strategy = best_strategy_row['strategy']
    best_score = best_strategy_row['score']
    summary_lines.append(f"\nBest Performing Strategy: {best_strategy} (Average Score: {best_score:.3f})\n")
    logger.info(f"Best Performing Strategy: {best_strategy} with Average Score: {best_score:.3f}")

    summary_lines.append("Top Performing Strategies per Metric")
    summary_lines.append("-" * 40)

    # For each metric, find the strategy with the highest average score
    metrics = metrics_df['metric'].unique()
    for metric in metrics:
        metric_df = metrics_df[metrics_df['metric'] == metric]
        if metric_df.empty:
            continue
        top_strategy = metric_df.groupby('strategy')['score'].mean().idxmax()
        top_score = metric_df.groupby('strategy')['score'].mean().max()
        summary_str = (
            f"{metric}:\n"
            f"  Best Strategy: {top_strategy}\n"
            f"  Average Score: {top_score:.3f}\n"
        )
        summary_lines.append(summary_str)
        logger.info(
            f"Metric: {metric} | Best Strategy: {top_strategy} | "
            f"Average Score: {top_score:.3f}"
        )

    summary_path = output_dir / SUMMARY_REPORT_NAME
    try:
        with summary_path.open('w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        logger.info(f"Saved analysis summary to {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save summary report: {e}")
        raise


def main() -> None:
    """
    Main function to orchestrate the visualization and analysis of LLM responses.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Evaluation Visualization Tool")
    parser.add_argument(
        '--evaluation_file',
        type=str,
        default=None,
        help='Path to the evaluation results JSON file.'
    )
    args = parser.parse_args()

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime(DATE_FORMAT)
    output_dir = Path(OUTPUT_DIR_TEMPLATE.format(timestamp=timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting RAG evaluation visualization")

    # Define the path to the evaluation results JSON file
    if args.evaluation_file:
        evaluation_file = Path(args.evaluation_file)
    else:
        # Use default template with current timestamp
        evaluation_file = Path(
            DEFAULT_EVALUATION_FILE_TEMPLATE.format(timestamp=timestamp)
        )

    try:
        # Load evaluation results
        all_results = load_evaluation_results(evaluation_file, logger)

        if not all_results:
            logger.error("No evaluation results found.")
            sys.exit(1)

        # Create metrics DataFrame
        metrics_df = create_metrics_dataframe(all_results, logger)

        # Save metrics to CSV
        save_metrics_csv(metrics_df, output_dir, logger)

        # Generate visualizations
        plot_metrics_heatmap(metrics_df, output_dir, logger)
        plot_strategy_performance(metrics_df, output_dir, logger)
        plot_metric_comparison(metrics_df, output_dir, logger)
        plot_latency_analysis(all_results, output_dir, logger)

        # Generate summary report
        generate_summary_report(all_results, metrics_df, output_dir, logger)

        logger.info(f"Visualization results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
