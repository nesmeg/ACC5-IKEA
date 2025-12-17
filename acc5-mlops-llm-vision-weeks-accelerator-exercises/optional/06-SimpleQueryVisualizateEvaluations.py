import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime


# Constants
EVALUATION_FILE_TEMPLATE = "evaluation_results/strategies_evaluation_20250117_220939.json"
OUTPUT_DIR_TEMPLATE = "evaluation_results/visualization_{timestamp}"
LOG_FILE_NAME = "evaluation.log"
METRICS_CSV_NAME = "metrics_results.csv"
SUMMARY_REPORT_NAME = "analysis_summary.txt"
DATE_FORMAT = "%Y%m%d_%H%M%S"


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


def load_evaluation_results(file_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load the evaluation results from a JSON file.

    Args:
        file_path (Path): Path to the JSON file containing evaluation results.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        Dict[str, Any]: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    logger.info(f"Loading evaluation results from {file_path}")
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("Successfully loaded evaluation results")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading evaluation results: {e}")
        raise


def create_metrics_dataframe(results: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Create a DataFrame with all metrics for each strategy.

    Args:
        results (Dict[str, Any]): Evaluation results data.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: DataFrame containing metrics data.
    """
    logger.info("Creating metrics DataFrame")
    data = [
        {
            'strategy': strategy,
            'metric': metric,
            'score': details['score'],
            'reason': details['reason']
        }
        for strategy, strategy_details in results.get('strategies', {}).items()
        for metric, details in strategy_details.get('metrics', {}).items()
    ]
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

    pivot_data = df.pivot(index='strategy', columns='metric', values='score')

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
            colorbar=dict(title='Score')
        )
    )

    fig.update_layout(
        title='RAG Strategy Performance Across Metrics',
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
    Create an interactive bar plot with error bars.

    Args:
        df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the plot will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Creating strategy performance plot")

    strategy_stats = df.groupby('strategy')['score'].agg(['mean', 'std']).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Score',
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
    Create an interactive grouped bar plot.

    Args:
        df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the plot will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Creating metric comparison plot")

    try:
        fig = px.bar(
            df,
            x='strategy',
            y='score',
            color='metric',
            barmode='group',
            title='Metric Scores by Strategy',
            labels={'strategy': 'Strategy', 'score': 'Score', 'metric': 'Metric'},
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

        fig.write_html(output_dir / 'metric_comparison.html')
        fig.write_image(output_dir / 'metric_comparison.png')
        logger.info("Saved metric comparison visualizations")
    except Exception as e:
        logger.error(f"Failed to save metric comparison visualizations: {e}")
        raise


def plot_latency_analysis(results: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> None:
    """
    Create interactive scatter plots analyzing performance metrics.

    Args:
        results (Dict[str, Any]): Evaluation results data.
        output_dir (Path): Directory where the plots will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Creating latency analysis plots")

    summary = results.get('summary', {})
    if not summary:
        logger.warning("No summary data available for latency analysis")
        return

    data = pd.DataFrame([
        {
            'strategy': strategy,
            'latency': details.get('latency', np.nan),
            'average_score': details.get('average_score', np.nan)
        }
        for strategy, details in summary.items()
    ])

    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=('Latency vs Performance',)
    )

    fig.add_trace(
        go.Scatter(
            x=data['latency'],
            y=data['average_score'],
            mode='markers+text',
            text=data['strategy'],
            textposition="top center",
            marker=dict(size=10, color='blue', opacity=0.7),
            name='Latency'
        ),
        row=1,
        col=1
    )

    fig.update_layout(
        title='Latency vs. Average Performance',
        height=600,
        width=800,
        showlegend=False
    )

    fig.update_xaxes(title_text="Latency (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Average Score", row=1, col=1)

    try:
        fig.write_html(output_dir / 'performance_latency_analysis.html')
        fig.write_image(output_dir / 'performance_latency_analysis.png')
        logger.info("Saved latency analysis visualizations")
    except Exception as e:
        logger.error(f"Failed to save latency analysis visualizations: {e}")
        raise


def generate_summary_report(
    results: Dict[str, Any],
    metrics_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Generate a detailed text summary of the analysis.

    Args:
        results (Dict[str, Any]): Evaluation results data.
        metrics_df (pd.DataFrame): DataFrame containing metrics data.
        output_dir (Path): Directory where the summary report will be saved.
        logger (logging.Logger): Logger instance for logging messages.
    """
    logger.info("Generating summary report")

    summary_lines = [
        "RAG Strategy Evaluation Summary",
        "=" * 30 + "\n"
    ]

    query = results.get('query', 'N/A')
    summary_lines.append(f"Query: {query}\n")
    logger.info(f"Analyzing results for query: {query}")

    summary_lines.append("Overall Performance")
    summary_lines.append("-" * 20)
    summary_lines.append("")

    summary = results.get('summary', {})
    for strategy, details in summary.items():
        average_score = details.get('average_score', 0.0)
        latency = details.get('latency', 0.0)
        context_size = details.get('context_size', 'N/A')

        performance_str = (
            f"{strategy}:\n"
            f"  Average Score: {average_score:.3f}\n"
            f"  Latency: {latency:.3f} seconds\n"
            f"  Context Size: {context_size}"
        )
        summary_lines.append(performance_str)
        summary_lines.append("")

        logger.info(
            f"Performance metrics for {strategy}: "
            f"score={average_score:.3f}, latency={latency:.3f}s, "
            f"context_size={context_size}"
        )

    if summary:
        avg_scores = {
            s: d.get('average_score', 0.0) for s, d in summary.items()
        }
        best_strategy = max(avg_scores, key=avg_scores.get)
        best_score = avg_scores[best_strategy]
        summary_lines.append(f"\nBest Performing Strategy: {best_strategy}")
        summary_lines.append(f"Score: {best_score:.3f}")
        logger.info(f"Best strategy: {best_strategy} with score {best_score:.3f}")

    summary_lines.append("\nMetric Analysis")
    summary_lines.append("-" * 15)

    for metric in metrics_df['metric'].unique():
        metric_scores = metrics_df[metrics_df['metric'] == metric]
        if metric_scores.empty:
            continue
        best_metric_row = metric_scores.loc[metric_scores['score'].idxmax()]
        metric_summary = (
            f"\n{metric}:\n"
            f"  Best Strategy: {best_metric_row['strategy']}\n"
            f"  Score: {best_metric_row['score']:.3f}\n"
            f"  Reason: {best_metric_row['reason']}"
        )
        summary_lines.append(metric_summary)
        logger.info(
            f"Best strategy for {metric}: {best_metric_row['strategy']} "
            f"with score {best_metric_row['score']:.3f}"
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
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime(DATE_FORMAT)
    output_dir = Path(OUTPUT_DIR_TEMPLATE.format(timestamp=timestamp))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting RAG evaluation visualization")

    # Define the path to the evaluation results JSON file
    evaluation_file = Path(
        EVALUATION_FILE_TEMPLATE.format(timestamp=timestamp)
    )

    try:
        # Load evaluation results
        results = load_evaluation_results(evaluation_file, logger)

        # Create metrics DataFrame
        metrics_df = create_metrics_dataframe(results, logger)

        # Save metrics to CSV
        save_metrics_csv(metrics_df, output_dir, logger)

        # Generate visualizations
        plot_metrics_heatmap(metrics_df, output_dir, logger)
        plot_strategy_performance(metrics_df, output_dir, logger)
        plot_metric_comparison(metrics_df, output_dir, logger)
        plot_latency_analysis(results, output_dir, logger)

        # Generate summary report
        generate_summary_report(results, metrics_df, output_dir, logger)

        logger.info(f"Visualization results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
