"""Utility functions for microbiome-ml."""

import argparse
import logging
import os

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Parameters
    ----------
    name : str
        Name for the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    """
    # Create a custom logger
    logger = logging.getLogger(name)

    # Set the default logging level
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()

    # Set levels for handlers
    c_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    # Add the handler to the logger
    logger.addHandler(c_handler)

    return logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the microbiome ML pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(description="Microbiome ML Pipeline")
    parser.add_argument(
        "--abundance", type=str, required=True, help="Path to abundance data file"
    )
    parser.add_argument(
        "--metadata", type=str, required=True, help="Path to metadata file"
    )
    parser.add_argument(
        "--target", type=str, default="Group", help="Target variable in metadata"
    )
    parser.add_argument(
        "--sample_column_name",
        type=str,
        default="Sample ID",
        help="Column name for sample IDs in metadata",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for saving results",
    )
    args = parser.parse_args()
    return args


def create_output_dir(output_dir: str, logger: logging.Logger | None = None) -> None:
    """Ensure the output directory exists.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    logger : logging.Logger, optional
        Logger for info messages (default: module logger).

    """
    if logger is None:
        logger = get_logger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created or already exists: {output_dir}")


def save_probabilities(
    output_dir: str,
    clr_data: pd.DataFrame,
    probs: pd.Series,
    logger: logging.Logger | None = None,
) -> None:
    """Save predicted probabilities to a CSV file.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    clr_data : pd.DataFrame
        DataFrame with sample indices.
    probs : pd.Series
        Predicted probabilities for each sample.
    logger : logging.Logger, optional
        Logger for info messages (default: module logger).

    """
    if logger is None:
        logger = get_logger(__name__)
    probs_df = pd.DataFrame(probs, index=clr_data.index, columns=["Probabilities"])
    probs_file_path = os.path.join(output_dir, "predicted_probabilities.csv")
    probs_df.to_csv(probs_file_path, index=True)
    logger.info(f"Predicted probabilities saved to {probs_file_path}")


def save_results(
    output_dir: str,
    clr_data: pd.DataFrame,
    probs: pd.Series,
    labels: pd.Series,
    logger: logging.Logger | None = None,
) -> None:
    """Save results and plots to the output directory.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    clr_data : pd.DataFrame
        CLR-transformed data.
    probs : pd.Series
        Predicted probabilities.
    labels : pd.Series
        True labels for the samples.
    logger : logging.Logger, optional
        Logger for info messages (default: module logger).

    """
    if logger is None:
        logger = get_logger(__name__)
    create_output_dir(output_dir, logger)
    save_probabilities(output_dir, clr_data, probs, logger)
