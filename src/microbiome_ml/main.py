"""Main CLI and pipeline entry point for microbiome-ml."""

import sys
import uuid

from microbiome_ml.data_processing import clr_transform, filter_data, load_data
from microbiome_ml.modeling import train_model
from microbiome_ml.utils import get_logger, parse_args, save_results
from microbiome_ml.visualisation import save_roc_curve

app_logger = get_logger(__name__)

# Constants
PREDICTED_PROBABILITIES_FILE = "predicted_probabilities.csv"
ROC_CURVE_FILE = "roc_curve.png"


def run_pipeline(
    abundance_path: str,
    metadata_path: str,
    sample_column: str,
    target_column: str,
    output_dir: str,
    logger=app_logger,
) -> None:
    """Run the microbiome ML pipeline."""
    logger.info("Loading data...")
    abundance, labels = load_data(
        abundance_file=abundance_path,
        metadata_file=metadata_path,
        sample_column=sample_column,
        target_column=target_column,
    )
    logger.info(
        f"Data loaded: {abundance.shape[0]} features, {abundance.shape[1]} samples"
    )

    logger.info("Filtering and transforming data...")
    filtered_data = filter_data(abundance)
    clr_data = clr_transform(filtered_data)

    logger.info("Training the model...")
    # Set n_jobs=1 to avoid multiprocessing issues
    model = train_model(clr_data, labels, n_jobs=1)

    logger.info("Predicting probabilities...")
    probs = model.predict_proba(clr_data)[:, 1]

    if output_dir:
        output_dir = f"{output_dir}/{uuid.uuid4()}"
        save_results(output_dir, clr_data, probs, labels, logger)
        save_roc_curve(output_dir, labels, probs, logger)
        logger.info(f"Results saved to {output_dir}. ")

    logger.info("Pipeline completed successfully.")


def main() -> None:
    """Run the main entry point for the pipeline."""
    try:
        args = parse_args()
        run_pipeline(
            abundance_path=args.abundance,
            sample_column=args.sample_column_name,
            metadata_path=args.metadata,
            target_column=args.target,
            output_dir=args.output_dir,
        )
    except Exception as e:
        app_logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
