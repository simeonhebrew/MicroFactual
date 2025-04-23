import logging
import argparse
import os

import pandas as pd
    
def get_logger(name):
    """Get a logger with the specified name."""

    # Create a custom logger
    logger = logging.getLogger(name)

    # Set the default logging level
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()

    # Set levels for handlers
    c_handler.setLevel(logging.INFO)

    # Create formatters and add them to the handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    # Add the handler to the logger
    logger.addHandler(c_handler)

    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Microbiome ML Pipeline")
    parser.add_argument('--abundance', type=str, required=True, help='Path to abundance data file')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--target', type=str, default='Group', help='Target variable in metadata')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory for saving results')
    args = parser.parse_args()
    return args

def create_output_dir(output_dir: str, logger=get_logger(__name__)) -> None:
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created or already exists: {output_dir}")
    
def save_probabilities(output_dir: str, clr_data: pd.DataFrame, probs: pd.Series, logger=get_logger(__name__)) -> None:
    """Save predicted probabilities to a CSV file."""
    probs_df = pd.DataFrame(probs, index=clr_data.index, columns=['Probabilities'])
    probs_file_path = os.path.join(output_dir, 'predicted_probabilities.csv')
    probs_df.to_csv(probs_file_path, index=True)
    logger.info(f"Predicted probabilities saved to {probs_file_path}")
    
def save_results(output_dir: str, clr_data: pd.DataFrame, probs: pd.Series, labels: pd.Series, logger=get_logger(__name__)) -> None:
    """Save results and plots to the output directory."""
    create_output_dir(output_dir, logger)
    save_probabilities(output_dir, clr_data, probs, logger)
